#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

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

typedef struct board {
    struct row {
        // Each row is 21 entries wide to allow for edges.
        // Values defined above
        uint8_t s[21];
    };
    row rows[21];
    uint8_t flags;
} Board;

#define STONE_AT(b, r, c) ((b)->rows[r].s[c])
#define SET_STONE_AT(b, r, c, v) ((b)->rows[r].s[c] = v)

#define OPPOSITE(color) ((color == WHITE) ? BLACK : WHITE)

// XXX - TODO:
// - ko detection - easy? : if we remove one stone, and the played stone is in atari, removed space is ko?
// - is there an easy way to avoid playing in single eyes that are controlled by other player?
//    Maybe: 
//        add WHITE_PERMANENT and BLACK_PERMANENT: if alive and has two real single-space eyes.
//        don't play in single spaces surrounded by PERMANENTs.
       

// SINGLE EYE == 4 horizontal & vertical neighbors are all the right color, or edge.
#define COLOR_OR_EDGE_as_1(b, r, c, color) (((STONE_AT(b, r, c) == color) ? 1 : 0) +  \
                                            ((STONE_AT(b, r, c) == EDGE) ? 1 : 0))

#define SINGLE_EYE(b, r, c, color) ((COLOR_OR_EDGE_as_1(b, r + 1, c, color) +     \
                                     COLOR_OR_EDGE_as_1(b, r - 1, c, color) +     \
                                     COLOR_OR_EDGE_as_1(b, r, c + 1, color) +     \
                                     COLOR_OR_EDGE_as_1(b, r, c - 1, color)) == 4)


// FALSE_EYE is only valid if SINGLE_EYE is true
// - >= two diagonal neighbors are opposite color, or
// - 1 diagonal neighbor opposite color and at edge
#define DIAG_NEIGHBORS(b, r, c, color) (((STONE_AT(b, r + 1, c + 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r + 1, c - 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r - 1, c + 1) == color) ? 1 : 0) + \
                                        ((STONE_AT(b, r - 1, c - 1) == color) ? 1 : 0))

#define NEXT_TO(b, r, c, v)  ((STONE_AT(b, r + 1, c) == (v)) || \
                              (STONE_AT(b, r - 1, c) == (v)) || \
                              (STONE_AT(b, r, c + 1) == (v)) || \
                              (STONE_AT(b, r, c - 1) == (v)))

#define AT_EDGE(b, r, c) NEXT_TO(b, r, c, EDGE)

#define FALSE_EYE(b, r, c, color)  ((DIAG_NEIGHBORS(b, r, c, OPPOSITE(color)) >= 2) || \
                                    ((DIAG_NEIGHBORS(b, r, c, OPPOSITE(color)) == 1) && AT_EDGE(b, r, c)))


// Real single eyes = single eye, and not false.  Fails for some cases
// (see Two-headed dragon @ sensei's library).
#define SINGLE_REAL_EYE(b, r, c, color) (SINGLE_EYE(b, r, c, color) && (! FALSE_EYE(b, r, c, color)))


#define ALIVE(b, row, c, alive_color) (NEXT_TO(b, row, c, EMPTY) || NEXT_TO(b, row, c, alive_color))


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
    }
}

__device__ void remove_dead_groups(Board *b,
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

    if (row == 1) {
        if (num_changes > 0)
            printf("    removed %d stones\n", num_changes);
    }
}

__global__ void make_random_move(Board *b,
                                 uint8_t color,
                                 curandState *randstate)
{
    // NB: we add because interior space on board is 1 indexed
    int row = threadIdx.x + 1; 

    // local values
    int num_valid_moves = 0;
    int valid_move_mask = 0;
    int which_row = 1;
    int which_move = 0;

    // shared values
    __shared__ int thread_valid_moves[20];


    // remember 1-indexed because of edges, and see NB above
    if (row > 19) return;



    // **************************************************
    // FIND VALID PLAY LOCATIONS
    // **************************************************
    for (int c = 1; c <= 19; c++) {
        if ((STONE_AT(b, row, c) == EMPTY) && (! SINGLE_REAL_EYE(b, row, c, color))) {
            valid_move_mask |= (1 << c);
            num_valid_moves++;
        }
    }
    thread_valid_moves[row] = num_valid_moves;

    // figure out how many valid moves there were in the whole board
    num_valid_moves += __shfl_down(num_valid_moves, 16);
    num_valid_moves += __shfl_down(num_valid_moves, 8);
    num_valid_moves += __shfl_down(num_valid_moves, 4);
    num_valid_moves += __shfl_down(num_valid_moves, 2);
    num_valid_moves += __shfl_down(num_valid_moves, 1);

    // update all threads about valid move count
    num_valid_moves = __shfl(num_valid_moves, 0);

    if (num_valid_moves == 0)
        goto end;


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
        if (__popc(valid_move_mask) != thread_valid_moves[which_row])
            printf("BAD POPC\n");
        if (which_move >= thread_valid_moves[which_row])
            printf("BAD WHICH MOVE\n");

        // find which column to place move at
        int which_col = 1;
        do {
            // shift which_col to the next set bit in valid_move_mask
            while (! (valid_move_mask & (1 << which_col)))
                which_col++;
            if (which_move > 0)
                which_col++;
            which_move--;
        } while (which_move >= 0);

        printf("%d total valid moves\n", num_valid_moves);
        printf("    placed at %d %d\n", which_row, which_col);
        SET_STONE_AT(b, which_row, which_col, color);
    }

    // **************************************************
    // REMOVE DEAD GROUPS
    // **************************************************
    remove_dead_groups(b, OPPOSITE(color));
    remove_dead_groups(b, color);

 end:
    return;
}

__global__ void setup_random(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64; 
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed ^ id, id, 0, &state[id]);
}

int main(void)
{
    void *cuboard;
    Board board;
    curandState *randstates;

    cudaMalloc(&cuboard, sizeof (Board));
    cudaMalloc(&randstates, 10 * sizeof(curandState));

    setup_random<<<1, 10>>>(randstates);
    clear_board<<<1, 32>>>((Board *) cuboard);

    for (int i = 0; i < 500; i++) {
        make_random_move<<<1, 32>>>((Board *) cuboard, BLACK, randstates);
        make_random_move<<<1, 32>>>((Board *) cuboard, WHITE, randstates);
    }

    cudaMemcpy(&board, cuboard, sizeof (Board), cudaMemcpyDeviceToHost);

    for(int i=0; i < 21; i++) {
        printf ("%02d ", i);
        for(int j=0; j < 21; j++) {
            char c = stone_chars[STONE_AT(&board, i, j)];
            if (STONE_AT(&board, i, j) == EMPTY) {
                if (SINGLE_EYE(&board, i, j, WHITE)) 
                    if (FALSE_EYE(&board, i, j, WHITE)) {
                        assert(! SINGLE_REAL_EYE(&board, i, j, WHITE));
                        c = 'F';
                    } else {
                        assert(SINGLE_REAL_EYE(&board, i, j, WHITE));
                        c = 'E';
                    }
                if (SINGLE_EYE(&board, i, j, BLACK))
                    if (FALSE_EYE(&board, i, j, BLACK)) {
                        assert(! SINGLE_REAL_EYE(&board, i, j, BLACK));
                        c = 'F';
                    } else {
                        assert(SINGLE_REAL_EYE(&board, i, j, BLACK));
                        c = 'E';
                    }
            }
            printf(" %c", c);
        }

        printf("\n");
    }

    cudaDeviceReset();
    return 0;
}
