#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

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
    uint8_t flags;
    uint8_t ko_row, ko_col;
} Board;

#define STONE_AT(b, r, c) ((b)->cols[c].rows[r])
#define SET_STONE_AT(b, r, c, v) ((b)->cols[c].rows[r] = v)

#define OPPOSITE(color) ((color == WHITE) ? BLACK : WHITE)

// XXX - TODO:
// - write scoring function.
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
    if (row < 21) {
        SET_STONE_AT(b, row, 0, EDGE);
        for (int c = 1; c <= 19; c++)
            SET_STONE_AT(b, row, c, ((row == 0) || (row == 20)) ? EDGE : EMPTY);
        SET_STONE_AT(b, row, 20, EDGE);
    }
    if (row == 0) {
        b->flags = 0;
        b->ko_row = 0;
    }
}

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
        for (int c = 1; c <= 19; c++) {
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
    for (int c = 1; c <= 19; c++) {
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

    // update all threads about total removed
    num_changes = __shfl(num_changes, 0);

    if (LOG && (row == 1)) {
        if (num_changes > 0)
            printf("    removed %d %s stones\n", num_changes, NAME(color));
    }

    // NB: all threads must return the same value for
    // make_random_move() to work correctly.
    return num_changes;
}

// **************************************************
// Makes a random move and returns true if the board changed.
// **************************************************
__device__ int make_random_move(Board *b,
                                uint8_t color,
                                curandState *randstate)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;

    // local values
    int num_valid_moves;
    int valid_move_mask;
    int num_killed;
    int num_suicide;

    // where the random move is made
    int which_move = 0;
    int which_row;
    int which_col;

    // shared values
    __shared__ int thread_valid_moves[20];


    // remember 1-indexed because of edges, and see NB above
    if (row > 19) return 0;

    // **************************************************
    // FIND VALID PLAY LOCATIONS
    // **************************************************
    valid_move_mask = 0;
    for (int c = 1; c <= 19; c++) {
        if ((STONE_AT(b, row, c) == EMPTY) && (! SINGLE_REAL_EYE(b, row, c, color))) {
            valid_move_mask |= (1 << c);
        }
    }

    // Disallow retaking the ko
    if (row == b->ko_row) {
        valid_move_mask &= ~ (1 << b->ko_col);
    }

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
    
    // **************************************************
    // REMOVE DEAD GROUPS & KO DETECTION
    // **************************************************
    num_killed = 0;
    num_suicide = 0;

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

    // NB: all threads will return the same value

    // Return whether the board changed.
    // The only way that didn't happen is if this was a single-stone suicide play.
    return (num_suicide != 1);
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
    curandState *my_rand = randstates + blockIdx.x;
    Board *my_board = boards + blockIdx.x;

    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1;

    if (row > 19) return;

    if (row == 1)
        *my_board = *start_board;

    while ((move_count < max_moves) && (unchanged_count < max_unchanged)) {
        move_count++;
        int board_changed = make_random_move(my_board, current_color, my_rand);
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

__global__ void setup_random(curandState *states)
{
    unsigned int id = blockIdx.x;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed ^ (id << 6), id, 0, &(states[id]));
}

#define COUNT 10000

int main(void)
{
    void *start_board, *playouts, *board_sum;
    Board board;
    curandState *randstates;
    cudaEvent_t start, end;

    cudaMalloc(&start_board, sizeof (Board));
    cudaMalloc(&playouts, COUNT * sizeof (Board));
    cudaMalloc(&board_sum, sizeof (Board));
    cudaMalloc(&randstates, COUNT * sizeof(curandState));

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    setup_random<<<COUNT, 1>>>(randstates);
    clear_board<<<1, 32>>>((Board *) start_board);
    cudaMemcpy(&board, (Board *) start_board, sizeof (Board), cudaMemcpyDeviceToHost);

    // Game #768554 - played out by gnugo.  3.5 at the end in chinese scoring.
    char game[] = "[pd][dp][pq][dd][qk][lp][cj][cl][cg][gc][lc][jc][le][qc][qd][pc][nc][od][oc][rd][re][rc][qe][oe][qg][og][lg][oi][qi][ok][cn][dk][cq][cp][dq][ep][eq][fq][fr][gq][gr][hq][bp][bo][bq][bn][be][nq][po][li][pb][qb][ob][kh][mn][on][np][op][oo][no][mp][mo][mq][mr][nn][lo][om][pl][pp][lq][ln][ql][pm][kn][km][jm][jl][kl][lm][im][jk][lk][hk][ih][fk][gl][gk][fl][fh][ej][gi][kg][ek][dj][el][dh][dg][eg][eh][ci][ef][ge][gf][fe][if][ff][fg][hf][hg][gg][hh][ig][cc][dc][cb][kb][lb][lf][mf][kf][mg][me][ne][md][nd][nf][ke][mh][ng][of][ld][nh][mc][em][dl][dm][id][ic][rk][je][jd][ie][hd][cd][bd][qm][qn][hc][kc][me][md][ja][sb][hr][er][or][pr][ps][qs][os][qr][sm][rn][db][da][ea][ca][fb][rl][rm][sn][bh][bg][hl][il][jn][ch][bj][di][bi][hs][is][gs][ij][ik][ir][la][lh][me][jj][nk][nj][mk][mj][ol][pk][pj][pi][ph][oh][qj][oj][ll][kk][ce][hj][gj][hi][gh][pg][qh][ag][af][ah][ee][ed][sl][ap][aq][ao][na][gd][ei][de][df][qf][rf][pf][ka][jb][fj][ck][oq][np][gf][he][kd][pa][mb][co]";
    int curcolor = BLACK;
    for (int idx = 1; idx < sizeof(game); idx += 4) {
        int c = game[idx] - 'a' + 1;
        int r = game[idx + 1] - 'a' + 1;
        SET_STONE_AT(&board, r, c, curcolor);
        curcolor = OPPOSITE(curcolor);
    }
    cudaMemcpy((Board *) start_board, &board, sizeof (Board), cudaMemcpyHostToDevice);
    
    
    cudaEventRecord(start, 0);
    play_out<<<COUNT, 32>>>((Board *) start_board, (Board *) playouts, 
                            BLACK, 1000, 100, randstates);

    sum_boards<<<1, dim3(19, 19)>>>((Board *) playouts, COUNT, (Board *) board_sum);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float delta_ms;
    cudaEventElapsedTime(&delta_ms, start, end);
    printf("%d boards in %0.2f ms\n", COUNT, delta_ms);


    cudaMemcpy(&board, ((Board *) playouts), sizeof (Board), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 21; i++) {
        for(int j = 0; j < 19; j++) {
            printf("%c ", stone_chars[STONE_AT(&board, i, j)]);
        }
        printf("\n");
    }
    printf("\n");


    cudaMemcpy(&board, ((Board *) board_sum), sizeof (Board), cudaMemcpyDeviceToHost);

    int total = 0;

    printf("[");
    for(int i = 0; i < 19; i++) {
        printf ("[");
        for(int j = 0; j < 19; j++) {
            printf("%d,", STONE_AT(&board, i + 1, j + 1));
            total += STONE_AT(&board, i + 1, j + 1);
        }
        printf("],\n");
    }
    printf("]\n");
    printf("expected score (B over W): %f\n", 2 * (total / ((float) COUNT)) - 361.0);

    cudaDeviceReset();
    return 0;
}
