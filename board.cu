#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>

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

const char stone_chars[] = ".BW#";

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

// XXX - TODO:
// - define SINGLE_REAL_EYE = SINGLE_EYE and (! FALSE_EYE) - see pachi's defn.
// - add dead group removal -- flood fill aliveness - how many iterations and how to parallelize?

#define SINGLE_REAL_EYE(b, r, c) (0)

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

    valid_move_mask = 0;

    for (int c = 1; c <= 19; c++) {
        if ((STONE_AT(b, row, c) == EMPTY) && (! SINGLE_REAL_EYE(b, row, c))) {
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

        printf("placed at %d %d\n", which_row, which_col);
        SET_STONE_AT(b, which_row, which_col, color);
    }

    if (row == 1) {
        printf("%d total valid moves\n", num_valid_moves);
    }
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

    for (int i = 0; i < 100; i++) {
        make_random_move<<<1, 32>>>((Board *) cuboard, BLACK, randstates);
        make_random_move<<<1, 32>>>((Board *) cuboard, WHITE, randstates);
    }

    cudaMemcpy(&board, cuboard, sizeof (Board), cudaMemcpyDeviceToHost);

    for(int i=0; i < 21; i++) {
        for(int j=0; j < 21; j++) {
            char c = stone_chars[STONE_AT(&board, i, j)];
            printf("%c", c);
        }

        printf("\n");
    }

    cudaDeviceReset();
    return 0;
}
