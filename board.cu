#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

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
        // Each col & row is 21 entries wide to allow for edges.
        // Values defined above
        uint32_t rows[21];
    };
    col cols[21];
    uint8_t size;
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


__global__ void clear_board(Board *b, int board_size)
{
    int row = threadIdx.x;
    assert (blockDim.x == board_size + 2);

    SET_STONE_AT(b, row, 0, EDGE);
    for (int c = 1; c <= board_size; c++)
        SET_STONE_AT(b, row, c, ((row == 0) || (row == board_size + 1)) ? EDGE : EMPTY);
    SET_STONE_AT(b, row, board_size + 1, EDGE);
    if (row == 0) {
        b->size = board_size;
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
    int num_changes;
    int alive_color = color + ALIVE_OFFSET;

    // Loop until no new updates have been made
    num_changes = 1;
    while (num_changes > 0) {
        num_changes = 0;
        for (int c = 1; c <= b->size; c++) {
            if ((STONE_AT(b, row, c) == color) && ALIVE(b, row, c, alive_color)) {
                SET_STONE_AT(b, row, c, alive_color);
                num_changes++;
            }
        }
        // figure out how many updates there were on the whole board
        num_changes += __shfl_down(num_changes, 16);
        num_changes += __shfl_down(num_changes, 8);
        num_changes += __shfl_down(num_changes, 4);
        num_changes += __shfl_down(num_changes, 2);
        num_changes += __shfl_down(num_changes, 1);

        // update all threads about total updates
        num_changes = __shfl(num_changes, 0);
    }

    // replace alive stones with stones of that color, and not-alive with empty.
    num_changes = 0;
    for (int c = 1; c <= b->size; c++) {
        if (STONE_AT(b, row, c) == color) {
            SET_STONE_AT(b, row, c, EMPTY);
            num_changes++;
        }
        else if (STONE_AT(b, row, c) == alive_color)
            SET_STONE_AT(b, row, c, color);
    }

    num_changes += __shfl_down(num_changes, 16);
    num_changes += __shfl_down(num_changes, 8);
    num_changes += __shfl_down(num_changes, 4);
    num_changes += __shfl_down(num_changes, 2);
    num_changes += __shfl_down(num_changes, 1);

    if (LOG && (row == 1)) {
        if (num_changes > 0)
            printf("    removed %d %s stones\n", num_changes, NAME(color));
    }

    // NB: all threads must return the same value for functions below
    // to work correctly.
    //
    // update all threads about total removed.
    num_changes = __shfl(num_changes, 0);
    return num_changes;
}

// **************************************************
// FIND VALID PLAY LOCATIONS
//
// return row-specific per-thread mask of valid moves
// **************************************************
__forceinline__ __device__ int find_valid_move_mask(Board *b, uint8_t color)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int valid_move_mask = 0;
    
    for (int c = 1; c <= b->size; c++) {
        if ((STONE_AT(b, row, c) == EMPTY) && (! SINGLE_REAL_EYE(b, row, c, color))) {
            valid_move_mask |= (1 << c);
        }
    }

    // Disallow retaking the ko
    if (row == b->ko_row) {
        valid_move_mask &= ~ (1 << b->ko_col);
    }

    return valid_move_mask;
}

// **************************************************
// REPORT VALID PLAY LOCATIONS
//
// set a board to BLACK where moves can be played
// **************************************************
__global__ void compute_valid_moves_board(Board *b_in, Board *b_out, uint8_t color)
{
    int row = threadIdx.x + 1;
    assert (blockDim.x == b_in->size);

    int valid_move_mask = find_valid_move_mask(b_in, color);

    for (int col = 1; col <= b_in->size; col++) {
        SET_STONE_AT(b_out, row, col, (valid_move_mask & (1 << col)) ? BLACK : EMPTY);
    }
}


// **************************************************
// REMOVE DEAD GROUPS & KO DETECTION
// 
// returns whether the board changed
// **************************************************
__forceinline__ __device__ int find_dead_groups(Board *b, int which_row, int which_col, uint8_t color, const int row)
{
    int num_killed = 0;
    int num_suicide = 0;

    if (IS_NEXT_TO(b, which_row, which_col, OPPOSITE(color)))
        num_killed = remove_dead_groups(b, OPPOSITE(color));

    // only check for suicide moves if necessary
    if ((num_killed == 0) && (! IS_NEXT_TO(b, which_row, which_col, EMPTY)))
        num_suicide = remove_dead_groups(b, color);

    if (row == 1) {
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

    // Return whether the board changed.
    // The only way that didn't happen is if this was a single-stone suicide play.
    // NB: all threads will return the same value due to 
    return (num_suicide != 1);
}


// **************************************************
// Makes a random move and returns true if the board changed.
// **************************************************
__device__ __inline__ int make_random_move(Board *b,
                                           uint8_t color,
                                           curandState *randstate,
                                           const int row)
{
    // local values
    int num_valid_moves;
    int valid_move_mask;

    // where the random move is made
    int which_move = 0;
    int which_row;
    int which_col;

    // shared values
    __shared__ int thread_valid_moves[20];

    valid_move_mask = find_valid_move_mask(b, color);

    // **************************************************
    // COUNT VALID MOVES
    // **************************************************
    num_valid_moves = __popc(valid_move_mask);
    thread_valid_moves[row] = num_valid_moves;

    // figure out how many valid moves there were in the whole board
    num_valid_moves += __shfl_down(num_valid_moves, 16);
    num_valid_moves += __shfl_down(num_valid_moves, 8);
    num_valid_moves += __shfl_down(num_valid_moves, 4);
    num_valid_moves += __shfl_down(num_valid_moves, 2);
    num_valid_moves += __shfl_down(num_valid_moves, 1);

    // update all threads about valid move count
    num_valid_moves = __shfl(num_valid_moves, 0);

    if (num_valid_moves == 0) {
        // forced pass

        // clear ko flag
        if (row == 1)
            b->ko_row = 0;
        return 0;  // no change in board
    }

    // **************************************************
    // CHOOSE RANDOM ROW BASED ON VALID MOVE COUNTS
    // **************************************************
    if (row == 1) {
        // choose one row to execute a move
        which_move = curand(randstate) % num_valid_moves;
        which_row = 1;
        while (which_move >= thread_valid_moves[which_row]) {
            which_move -= thread_valid_moves[which_row];
            which_row++;
        }
    }

    // all threads have to execute the shuffle
    valid_move_mask = __shfl(valid_move_mask, which_row - 1);

    // **************************************************
    // MAKE RANDOM MOVE IN CHOSEN ROW
    // **************************************************
    if (row == 1) {
        // find which column to place move at
        which_col = 1;
        do {
            // shift which_col to the next set bit in valid_move_mask
            while (! (valid_move_mask & (1 << which_col)))
                which_col++;
            if (which_move > 0)
                which_col++;
            which_move--;
        } while (which_move >= 0);
        if (LOG) {
            printf("%d total valid moves\n", num_valid_moves);
            printf("    placed %s at %d %d\n", NAME(color), which_row, which_col);
        }
        SET_STONE_AT(b, which_row, which_col, color);
    }

    // update all threads about where we played
    which_row = __shfl(which_row, 0);
    which_col = __shfl(which_col, 0);

    // NB: all threads will return the same value
    return find_dead_groups(b, which_row, which_col, color, row);
}

__global__ void play_moves(Board *b, char *moves)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    assert (blockDim.x == b->size);

    int idx = 0;
    while (moves[idx] != '\0') {
        if (LOG && (row == 1))
            printf("move: %c%c%c\n", moves[idx], moves[idx+1], moves[idx+2]);
        uint32_t color = (moves[idx] == 'B') ? BLACK : WHITE;
        int which_col = moves[idx + 1] - 'a' + 1;
        int which_row = moves[idx + 2] - 'a' + 1;
        // passes are encoded as moves outside the board
        if ((which_row == row) && (which_col <= b->size)) {
            if (LOG)
                printf("moved %s at %d %d\n", NAME(color), which_row, which_col);
            assert (STONE_AT(b, which_row, which_col) == EMPTY);
            SET_STONE_AT(b, which_row, which_col, color);
        }
        find_dead_groups(b, which_row, which_col, color, row);
        idx += 3;
    }
}

__global__ void make_one_move(Board *b_in, Board *b_out,
                              int which_row, int which_col,
                              uint8_t color)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    assert (blockDim.x == b_in->size);

    if (row == 1)
        *b_out = *b_in;

    if (which_row == row) {
        assert (STONE_AT(b_out, which_row, which_col) == EMPTY);
        SET_STONE_AT(b_out, which_row, which_col, color);
    }
    find_dead_groups(b_out, which_row, which_col, color, row);
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

    assert (blockDim.x == start_board->size);

    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;

    if (row == 1)
        *my_board = *start_board;

    while ((move_count < max_moves) && (unchanged_count < max_unchanged)) {
        move_count++;
        int board_changed = make_random_move(my_board, current_color, randstates + blockIdx.x, row);
        unchanged_count = board_changed ? 0 : (unchanged_count + 1);
        current_color = OPPOSITE(current_color);
        if (LOG && (row == 1))
            printf("unchanged: %d, move_count: %d\n", unchanged_count, move_count);
    }
}

__global__ void sum_boards(Board *start_board,
                           int num_boards,
                           Board *dest_board)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;
    if ((row > 19) || (col > 19))
        return;

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

    assert (blockDim.x == my_board->size);

    int count = 0;

    for (int col = 1; col <= my_board->size; col++) {
        int color = STONE_AT(my_board, row, col);
        if ((color == BLACK) || ((color == EMPTY) &&
                                 IS_NEXT_TO(my_board, row, col, BLACK)))
            count++;
        else 
            count--;
    }
    count += __shfl_down(count, 16);
    count += __shfl_down(count, 8);
    count += __shfl_down(count, 4);
    count += __shfl_down(count, 2);
    count += __shfl_down(count, 1);

    // code above is for calculating B - W, so subtract komi
    if (row == 1)
        results[blockIdx.x] = (count - komi) > 0;
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

__global__ void setup_random(curandState *states)
{
    unsigned int id = blockIdx.x;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed ^ (id << 6), id, 0, &(states[id]));
}

#define PLAYOUT_COUNT 1000

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

    int board_size = atoi(argv[1]);
    assert (board_size <= 19);
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
    clear_board<<<1, board_size + 2>>>((Board *) start_board, board_size);
    clear_board<<<1, board_size + 2>>>((Board *) moves_board, board_size);

    play_moves<<<1, board_size>>>((Board *) start_board, (char *)moves);

    // find valid moves
    compute_valid_moves_board<<<1, board_size>>>((Board *) start_board,
                                                 (Board *) moves_board,
                                                 color_to_play);

    // print board
    cudaMemcpy(&board, (Board *) start_board, sizeof (Board), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Generating move for %s on:\n", NAME(color_to_play));
    for(int i = 0; i < board_size + 2; i++) {
        for(int j = 0; j < board_size + 2; j++) {
            fprintf(stderr, "%c ", stone_chars[STONE_AT(&board, i, j)]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    
    // print valid locations
    cudaMemcpy(&board, (Board *) moves_board, sizeof (Board), cudaMemcpyDeviceToHost);

    fprintf(stderr, "valid moves:\n");
    for(int i = 0; i < board_size + 2; i++) {
        for(int j = 0; j < board_size + 2; j++) {
            fprintf(stderr, "%c ", stone_chars[STONE_AT(&board, i, j)]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    

    int num_boards_evaluated = 0;
    int best_win = 0;
    int best_i = -1;
    int best_j = -1;
    int moves_tested = 0;
    cudaEventRecord(start, 0);
    for(int i = 1; i <= board_size; i++) {
        for(int j = 1; j <= board_size; j++) {
            if (STONE_AT(&board, i, j) == BLACK) {
	        moves_tested++;
                make_one_move<<<1, board_size>>>((Board *) start_board, (Board *) next_board,
                                                 i, j, color_to_play);
                play_out<<<PLAYOUT_COUNT, board_size>>>((Board *) next_board, (Board *) playouts, 
                                                        OPPOSITE(color_to_play), 1000, 100, randstates);

                score_boards<<<PLAYOUT_COUNT, board_size>>>((Board *) playouts, komi, (int *) results);
                num_boards_evaluated++;
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
            }
        }
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

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
