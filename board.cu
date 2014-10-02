#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#ifndef BOARD_SIZE
#define BOARD_SIZE 9
#endif

#ifndef PLAYOUT_COUNT
#define PLAYOUT_COUNT 2500
#endif


#define LOG 0

// We want to be able to hold a board in shared memory

#define b00 0
#define b01 1
#define b10 2
#define b11 3

// Board values
#define EMPTY 0
#define BLACK 1
#define WHITE 2
#define EDGE  3
#define BLACK_ALIVE 4
#define WHITE_ALIVE 5

#define ALIVE_OFFSET 3

const char stone_chars[] = ".#o ";

#define NAME(color) ((color) == WHITE ? "white" : "black")

typedef struct board {
    struct col {
        // Each col & row is +2 entries wide to allow for edges.
        // Values defined above
        uint32_t rows[BOARD_SIZE + 2];
    };
    col cols[BOARD_SIZE + 2];
    uint8_t flags;
    uint8_t ko_row, ko_col;
} Board;

#define STONE_AT(b, r, c) ((b)->cols[c].rows[r])
#define SET_STONE_AT(b, r, c, v) ((b)->cols[c].rows[r] = v)

#define OPPOSITE(color) ((color == WHITE) ? BLACK : WHITE)

// XXX - TODO:
// - ko detection - testing.
// - is there an easy way to avoid playing in single eyes that are controlled by other player?
//    Maybe:
//        add WHITE_PERMANENT and BLACK_PERMANENT: if alive and has two real single-space eyes.
//        don't play in single spaces surrounded by PERMANENTs.


#define IS_NEXT_TO(b, r, c, v)  ((STONE_AT(b, r + 1, c) == (v)) || \
                                 (STONE_AT(b, r - 1, c) == (v)) || \
                                 (STONE_AT(b, r, c + 1) == (v)) || \
                                 (STONE_AT(b, r, c - 1) == (v)))

#define CT_NEXT_TO(b, r, c, v)  ((STONE_AT(b, r + 1, c) == (v)) + \
                                 (STONE_AT(b, r - 1, c) == (v)) + \
                                 (STONE_AT(b, r, c + 1) == (v)) + \
                                 (STONE_AT(b, r, c - 1) == (v)))


// SINGLE EYE == 4 horizontal & vertical neighbors are all the right color, or edge.
#define SINGLE_EYE(b, r, c, color) ((CT_NEXT_TO(b, r, c, color) + CT_NEXT_TO(b, r, c, EDGE)) == 4)


// FALSE_EYE is only valid if SINGLE_EYE is true
// - >= two diagonal neighbors are opposite color, or
// - 1 diagonal neighbor opposite color and at edge
#define DIAG_NEIGHBORS(b, r, c, color) (((STONE_AT(b, r + 1, c + 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r + 1, c - 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r - 1, c + 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r - 1, c - 1) == color) ? 1 : 0))

#define AT_EDGE(b, r, c) IS_NEXT_TO(b, r, c, EDGE)

#define FALSE_EYE(b, r, c, color)  ((DIAG_NEIGHBORS(b, r, c, OPPOSITE(color)) >= 2) || \
                                    ((DIAG_NEIGHBORS(b, r, c, OPPOSITE(color)) == 1) && AT_EDGE(b, r, c)))


// Real single eyes = single eye, and not false.  Fails for some cases
// (see Two-headed dragon @ sensei's library).
#define SINGLE_REAL_EYE(b, r, c, color) (SINGLE_EYE(b, r, c, color) && (! FALSE_EYE(b, r, c, color)))

#define ALIVE(b, row, c, alive_color) (IS_NEXT_TO(b, row, c, EMPTY) || IS_NEXT_TO(b, row, c, alive_color))

#define LONE_ATARI(b, row, c, color) ((CT_NEXT_TO(b, row, c, EMPTY) == 1) && (! (IS_NEXT_TO(b, row, c, color))))


__global__ void clear_board(Board *b)
{
    int row = threadIdx.x;
    assert (blockDim.x == BOARD_SIZE + 2);

    SET_STONE_AT(b, row, 0, EDGE);
    for (int c = 1; c <= BOARD_SIZE; c++)
        SET_STONE_AT(b, row, c, ((row == 0) || (row == BOARD_SIZE + 1)) ? EDGE : EMPTY);
    SET_STONE_AT(b, row, BOARD_SIZE + 1, EDGE);
    if (row == 0) {
        b->flags = 0;
        b->ko_row = 0;
    }
}

// **************************************************
// REMOVE DEAD GROUPS OF A GIVEN COLOR
// **************************************************
__device__ int remove_dead_groups(Board *b,
                                  uint8_t color)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    int changed;
    int alive_color = color + ALIVE_OFFSET;

    // Loop until no new updates have been made
    do {
        changed = 0;
        if ((STONE_AT(b, row, col) == color) && ALIVE(b, row, col, alive_color)) {
            SET_STONE_AT(b, row, col, alive_color);
            changed = 1;
        }
        // find if any boards changed across threads.
        changed = __syncthreads_or(changed);
    } while (changed);

    // replace alive stones with stones of that color, and not-alive with empty.
    changed = 0;
    if (STONE_AT(b, row, col) == color) {
        SET_STONE_AT(b, row, col, EMPTY);
        changed = 1;
    } else if (STONE_AT(b, row, col) == alive_color) {
        SET_STONE_AT(b, row, col, color);
    }

    // find how many stones died
    return __syncthreads_count(changed);
}
 
// **************************************************
// REMOVE DEAD GROUPS & KO DETECTION
// 
// returns whether the board changed
// **************************************************
__forceinline__ __device__ int find_dead_groups(Board *b, int which_row, int which_col, uint8_t color)
{
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    int num_killed = 0;
    int num_suicide = 0;


    // remove_dead_groups will change the board, so we need to
    // __syncthreads() between checking and calling.
    int check_for_dead = IS_NEXT_TO(b, which_row, which_col, OPPOSITE(color));
    __syncthreads(); 
    if (check_for_dead) num_killed = remove_dead_groups(b, OPPOSITE(color));

    // we don't need to syncthreads() here, because remove_dead_groups won't change EMPTYs.
    //
    // only check for suicide moves if necessary
    if ((num_killed == 0) && (! IS_NEXT_TO(b, which_row, which_col, EMPTY)))
        num_suicide = remove_dead_groups(b, color);

    // update ko state
    if ((row == which_row) && (col == which_col)) {
        if ((num_killed == 1) && LONE_ATARI(b, which_row, which_col, color)) {
            if      (STONE_AT(b, which_row + 1, which_col) == EMPTY) { b->ko_row = which_row + 1; b->ko_col = which_col; }
            else if (STONE_AT(b, which_row - 1, which_col) == EMPTY) { b->ko_row = which_row - 1; b->ko_col = which_col; }
            else if (STONE_AT(b, which_row, which_col + 1) == EMPTY) { b->ko_row = which_row;     b->ko_col = which_col + 1; }
            else if (STONE_AT(b, which_row, which_col - 1) == EMPTY) { b->ko_row = which_row;     b->ko_col = which_col - 1; }
            if (LOG) printf("     ko at %d %d\n", b->ko_row, b->ko_col);
         } else {
            b->ko_row = b->ko_col = 0;
        }
    }
    
    // sync to make sure ko state of board is updated.
    __syncthreads();

    // Return whether the board changed.
    // The only way that didn't happen is if this was a single-stone suicide play.
    // NB: all threads will return the same value.
    return (num_suicide != 1);
}


// **************************************************
// Makes a random move and returns true if the board changed.
// **************************************************
__device__ __inline__ int make_random_move(Board *b,
                                           uint8_t color,
                                           curandState *randstate)
{
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    // local values
    int total_valid;
    __shared__ int which_to_make, which_row, which_col;

    // index of thread in the current warp
    const int warpidx = (threadIdx.x + threadIdx.y * blockDim.x) % 32;

    // is this row/col a valid move?
    int is_valid = ((STONE_AT(b, row, col) == EMPTY) &&
                    (! SINGLE_REAL_EYE(b, row, col, color)) &&
                    ((b->ko_row != row) || (b->ko_col != col)));
    
    // get a count of number of valid moves in this warp
    int warp_valid_mask = __ballot(is_valid);
    int warp_valid_number = __popc(warp_valid_mask);

    // increment shared variable atomically, but only if at the head of a warp
    total_valid = __syncthreads_count(is_valid);

    // if there were no possible moves, return that the board has not changed.
    if (total_valid == 0) return 0;
    
    // have one thread choose a random value
    if ((row == 1) && (col == 1))
        which_to_make = curand(randstate) % total_valid;

    __syncthreads();  // wait for choice to be made

    // figure out which warp was chosen
    //
    // It's possible that warps could run in any order, but since we're
    // choosing a move randomly, it doesn't really matter.
    int this_warp_was_chosen = 0;
    int old;
    if (warpidx == 0) {
        // atomicSub returns the old value
        old = atomicSub(&which_to_make, warp_valid_number);
        this_warp_was_chosen = (old >= 0) && (old < warp_valid_number);
    }

    this_warp_was_chosen = __shfl(this_warp_was_chosen, 0);
    old = __shfl(old, 0);

    if (this_warp_was_chosen) {
        // find a mask for all bits below this one to apply to valid_move_mask
        unsigned int thread_below_mask = 1;
        thread_below_mask <<= warpidx;
        thread_below_mask -= 1;
    
        // if this square is a valid move, and the number of valid moves
        // below this thread in the warp == old, this is the square to
        // move at
        if ((is_valid) && (__popc(thread_below_mask & warp_valid_mask) == old)) {
            SET_STONE_AT(b, row, col, color);
            which_row = row;
            which_col = col;
        }
    }
    __syncthreads();

    // NB: all threads will return the same value
    return find_dead_groups(b, which_row, which_col, color);
}

__global__ void play_out(Board *start_board,
                         Board *boards,
                         uint8_t first_move_color,
                         int max_moves,
                         int max_unchanged,
                         curandState *randstates)
{
    int move_count = 0;
    int unchanged_count = 0;
    uint8_t current_color = first_move_color;
    Board *my_board = boards + blockIdx.x;

    assert (blockDim.x == BOARD_SIZE);
    assert (blockDim.y == BOARD_SIZE);

    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    if (row > BOARD_SIZE) return;
    if (col > BOARD_SIZE) return;

    if (row == 1 && col == 1)
        *my_board = *start_board;
    __syncthreads();

    while ((move_count < max_moves) && 
           (unchanged_count < max_unchanged)) {
        int board_changed = make_random_move(my_board, current_color, randstates + blockIdx.x);
        // There is a syncthreads() at all exits of make_random_move(), so no need to sync here.
        unchanged_count = board_changed ? 0 : (unchanged_count + 1);
        current_color = OPPOSITE(current_color);
        move_count++;
    }
}

__global__ void play_moves(Board *b, char *moves)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    assert (blockDim.x == BOARD_SIZE);
    assert (blockDim.y == BOARD_SIZE);

    int idx = 0;
    while (moves[idx] != '\0') {
        if (LOG && (row == 1) && (col == 1))
            printf("move: %c%c%c\n", moves[idx], moves[idx+1], moves[idx+2]);
        uint32_t color = (moves[idx] == 'B') ? BLACK : WHITE;
        int which_col = moves[idx + 1] - 'a' + 1;
        int which_row = moves[idx + 2] - 'a' + 1;
        // passes are encoded as moves outside the board
        if ((which_row == row) && (which_col == col)) {
            if (LOG)
                printf("moved %s at %d %d\n", NAME(color), which_row, which_col);
            assert (STONE_AT(b, which_row, which_col) == EMPTY);
            SET_STONE_AT(b, which_row, which_col, color);
        }
        __syncthreads();
        find_dead_groups(b, which_row, which_col, color);
        idx += 3;
    }
}

__global__ void make_one_move(Board *b_in, Board *b_out,
                              int which_row, int which_col,
                              uint8_t color)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;
    assert (blockDim.x == BOARD_SIZE);
    assert (blockDim.y == BOARD_SIZE);

    if ((row == 1) && (col == 1))
        *b_out = *b_in;

    __syncthreads();
    if ((which_row == row) && (which_col == col)) {
        assert (STONE_AT(b_out, which_row, which_col) == EMPTY);
        SET_STONE_AT(b_out, which_row, which_col, color);
    }
    __syncthreads();
    find_dead_groups(b_out, which_row, which_col, color);
}

__global__ void sum_boards(Board *start_board,
                           int num_boards,
                           Board *dest_board)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;
    assert (blockDim.x == BOARD_SIZE);
    assert (blockDim.y == BOARD_SIZE);

    int count = 0;

    for (int i = 0; i < num_boards; i++) {
        int color = STONE_AT(start_board + i, row, col);
        if ((color == BLACK) || ((color == EMPTY) &&
                                 IS_NEXT_TO(start_board + i, row, col, BLACK)))
            count++;
    }
    SET_STONE_AT(dest_board, row, col, count);
}

__global__ void score_boards(Board *boards,
                             float komi,
                             int *results)
{
    Board *my_board = boards + blockIdx.x;

    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;
    assert (blockDim.x == BOARD_SIZE);
    assert (blockDim.y == BOARD_SIZE);

    int color = STONE_AT(my_board, row, col);
    int black_count = __syncthreads_count((color == BLACK) || 
                                          ((color == EMPTY) && IS_NEXT_TO(my_board, row, col, BLACK)));
    int white_count = BOARD_SIZE * BOARD_SIZE - black_count;
    if ((row == 1) && (col == 1))
        results[blockIdx.x] = (black_count > white_count + komi);
}

__global__ void sum_results(int *results, int len)
{
    int sum = 0;
    int idx = threadIdx.x;
    assert (blockDim.x == 32);

    while (idx < len) {
        sum += results[idx];
        idx += 32;
    }
    // don't need a syncthreads, as we only have 32 threads.
    sum += __shfl_down(sum, 16);
    sum += __shfl_down(sum, 8);
    sum += __shfl_down(sum, 4);
    sum += __shfl_down(sum, 2);
    sum += __shfl_down(sum, 1);
    if (threadIdx.x == 0)
        results[0] = sum;
}

// **************************************************
// REPORT VALID PLAY LOCATIONS
//
// set a board to BLACK where moves can be played
// **************************************************
__global__ void compute_valid_moves_board(Board *b_in, Board *b_out, uint8_t color)
{
    int row = threadIdx.x + 1;
    assert (blockDim.x == BOARD_SIZE);

    for (int col = 1; col <= BOARD_SIZE; col++) {
        int valid = ((STONE_AT(b_in, row, col) == EMPTY) &&
                    (! SINGLE_REAL_EYE(b_in, row, col, color)) &&
                    ((b_in->ko_row != row) || (b_in->ko_col != col)));
        SET_STONE_AT(b_out, row, col, valid ? BLACK : EMPTY);
    }
}


__global__ void setup_random(curandState *states)
{
    unsigned int id = blockIdx.x;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed ^ (id << 6), id, 0, &(states[id]));
}

int main(int argc, char *argv[])
{
    void *start_board, *moves_board, *next_board, *playouts, *board_sum, *moves, *results;
    Board board;
    curandState *randstates;
    cudaEvent_t start, end;
    float delta_ms;

    // **************************************************
    // PARSE ARGUMENTS
    // **************************************************
    if (argc != 5) {
        printf("usage: board SIZE KOMI MOVES TO_PLAY\n");
        exit(1);
    }

    assert (atoi(argv[1]) == BOARD_SIZE);
    float komi = atof(argv[2]);
    char *_moves = argv[3];
    int color_to_play = (*argv[4] == 'W') ? WHITE : BLACK;

    // Copy moves to GPU
    cudaMalloc(&moves, strlen(_moves) + 1);
    cudaMemcpy((char *) moves, _moves, strlen(_moves) + 1, cudaMemcpyHostToDevice);

    // allocate other scratch space
    cudaMalloc(&start_board, sizeof (Board));
    cudaMalloc(&moves_board, sizeof (Board));
    cudaMalloc(&next_board, sizeof (Board));
    cudaMalloc(&playouts, PLAYOUT_COUNT * sizeof (Board));
    cudaMalloc(&board_sum, sizeof (Board));
    cudaMalloc(&results, PLAYOUT_COUNT * sizeof(int));
    cudaMalloc(&randstates, PLAYOUT_COUNT * sizeof(curandState));

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // initialize random states for each playout
    setup_random<<<PLAYOUT_COUNT, 1>>>(randstates);

    // clear and initialize boards
    clear_board<<<1, BOARD_SIZE + 2>>>((Board *) start_board);
    clear_board<<<1, BOARD_SIZE + 2>>>((Board *) moves_board);

    play_moves<<<1, dim3(BOARD_SIZE, BOARD_SIZE)>>>((Board *) start_board, (char *)moves);

    // find valid moves
    compute_valid_moves_board<<<1, BOARD_SIZE>>>((Board *) start_board,
                                                 (Board *) moves_board,
                                                 color_to_play);

    // print board
    cudaMemcpy(&board, (Board *) start_board, sizeof (Board), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Generating move for %s on:\n", NAME(color_to_play));
    for(int i = 0; i < BOARD_SIZE + 2; i++) {
        for(int j = 0; j < BOARD_SIZE + 2; j++) {
            fprintf(stderr, "%c ", stone_chars[STONE_AT(&board, i, j)]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    
    // print valid locations
    cudaMemcpy(&board, (Board *) moves_board, sizeof (Board), cudaMemcpyDeviceToHost);

    fprintf(stderr, "valid moves:\n");
    for(int i = 0; i < BOARD_SIZE + 2; i++) {
        for(int j = 0; j < BOARD_SIZE + 2; j++) {
            fprintf(stderr, "%c ", stone_chars[STONE_AT(&board, i, j)]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    

    int best_win = 0;
    int best_i = -1;
    int best_j = -1;
    int moves_tested = 0;
    cudaEventRecord(start, 0);
    for(int i = 1; i <= BOARD_SIZE; i++) {
        for(int j = 1; j <= BOARD_SIZE; j++) {
            if (STONE_AT(&board, i, j) == BLACK) {
	        moves_tested++;
                make_one_move<<<1, dim3(BOARD_SIZE, BOARD_SIZE)>>>((Board *) start_board, (Board *) next_board,
                                                                   i, j, color_to_play);

                play_out<<<PLAYOUT_COUNT, dim3(BOARD_SIZE, BOARD_SIZE)>>>((Board *) next_board, (Board *) playouts, 
                                                                                      OPPOSITE(color_to_play), 1000, 100, randstates);

                score_boards<<<PLAYOUT_COUNT, dim3(BOARD_SIZE, BOARD_SIZE)>>>((Board *) playouts, komi, (int *) results);

                sum_results<<<1, 32>>>((int *) results, PLAYOUT_COUNT);

                int win_count;
                cudaMemcpy(&win_count, results, sizeof(int), cudaMemcpyDeviceToHost);
                if (color_to_play == WHITE)
                    win_count = PLAYOUT_COUNT - win_count;
                fprintf(stderr, "at %d %d, %s wins %f\n", i, j,
                        NAME(color_to_play),
                        win_count / (float) PLAYOUT_COUNT);
                if (win_count > best_win) {
                    best_i = i;
                    best_j = j;
                    best_win = win_count;
                }
	        if (best_win > 0.95 * PLAYOUT_COUNT)
		  goto done;


                cudaError_t error = cudaGetLastError();
                if(error != cudaSuccess) {
                    printf("CUDA error: %s\n", cudaGetErrorString(error));
                    exit(-1);
                }
            }
        }
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&delta_ms, start, end);
    delta_ms /= moves_tested;
    fprintf(stderr, "%f msecs per 1K moves playouts\n", (delta_ms * PLAYOUT_COUNT) / 1000);
 done:
    if ((moves_tested > 0) && (best_win < 0.1 * PLAYOUT_COUNT)) {
        printf("resign");
    } else if (best_i > 0) {
        fprintf(stderr, "Playing at %d %d, expected win %f\n", best_i, best_j, best_win / (float) PLAYOUT_COUNT);
        printf("%d %d\n", best_j, best_i);
    } else {
        printf("pass\n");
    }

    cudaDeviceReset();
    exit(0);
}
