#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "cuda_inline_ptx.h"

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


// **************************************************
// Makes a random move and returns true if the board changed.
// **************************************************
__device__ __inline__ int make_random_move(Board *b, Board *b2,
                                           uint8_t color,
                                           curandState *randstate)
{
    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    // local values
    int total_valid;
    __shared__ int which_to_make;

    // get our laneid for some tricks below...
    const int laneid = __laneid();

    // is this row/col a valid move?
    int is_valid = ((STONE_AT(b, row, col) == EMPTY) &&
                    (! SINGLE_REAL_EYE(b, row, col, color)) &&
                    ((b->ko_row != row) || (b->ko_col != col)));
    
    // get a count of number of valid moves in this warp
    int warp_valid_mask = __ballot(is_valid);
    int warp_valid_count = __popc(warp_valid_mask);

    // find the total number of valid moves
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
    if (laneid == 0) { // only the first thread in a warp chooses
        // atomicSub returns the old value
        old = atomicSub(&which_to_make, warp_valid_count);
        this_warp_was_chosen = (old >= 0) && (old < warp_valid_count);
    }

    // tell our warp if we were chosen, and which of the active threads was chosen
    this_warp_was_chosen = __shfl(this_warp_was_chosen, 0);
    old = __shfl(old, 0);

    if (this_warp_was_chosen) {
        // find a mask for all bits below this one to apply to valid_move_mask
        unsigned int thread_below_mask = 1;
        thread_below_mask <<= laneid;
        thread_below_mask -= 1;
    
        // if this square is a valid move, and the number of valid moves
        // below this thread in the warp == old, this is the thread to move at
        if ((is_valid) && (__popc(thread_below_mask & warp_valid_mask) == old)) {
            SET_STONE_AT(b2, row, col, STONE_AT(b2, row, col) + 1);
        }
    }
    __syncthreads();

    return 0;
}


__global__ void play_random_moves()
{
    __shared__ Board b, b2;
    curandState state;

    unsigned int seed = (unsigned int) clock64();
    curand_init(seed, 0, 0, &state);

    int row = threadIdx.x + 1;
    int col = threadIdx.y + 1;

    if ((row == 1) && (col == 1)) {
        for (int r = 0; r < BOARD_SIZE + 2; r++) {
            SET_STONE_AT(&b, r, 0, EDGE);
            for (int c = 1; c <= BOARD_SIZE; c++)
                SET_STONE_AT(&b, r, c, ((r == 0) || (r == BOARD_SIZE + 1)) ? EDGE : EMPTY);
            SET_STONE_AT(&b, r, BOARD_SIZE + 1, EDGE);
            b.ko_row = 0;
        }

        SET_STONE_AT(&b, 4, 4, EDGE);
        SET_STONE_AT(&b, 4, BOARD_SIZE-4, EDGE);
        SET_STONE_AT(&b, BOARD_SIZE - 4, 4, EDGE);
        SET_STONE_AT(&b, BOARD_SIZE - 4, BOARD_SIZE - 4, EDGE);
        b2 = b;
    }
    __syncthreads();

    for (int i = 0; i < 100000; i++)
        make_random_move(&b, &b2, BLACK, &state);

    if ((row == 1) && (col == 1)) {
        printf("[");
        for (int r = 1; r < BOARD_SIZE + 1; r++) {
            printf("[");
            for (int c = 1; c < BOARD_SIZE + 1; c++)
                printf("%d, ", STONE_AT(&b2, r, c));
            printf("],\n");
        }
        printf("]\n");
    }
}

int main(int argc, char *argv[])
{

    play_random_moves<<<1, dim3(BOARD_SIZE, BOARD_SIZE)>>>();
    cudaDeviceSynchronize();
}
