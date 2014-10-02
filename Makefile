all: board19 board13 board9 test_random


board19: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=1000 -DBOARD_SIZE=19 -G -g -arch=sm_30 board.cu -o board19

board13: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=2500 -DBOARD_SIZE=13 -O3 -arch=sm_30 board.cu -o board13

board9: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=2500 -DBOARD_SIZE=9 -O3 -arch=sm_30 board.cu -o board9

test_random: test_random.cu
	/usr/local/cuda/bin/nvcc -DBOARD_SIZE=19 -O3 -arch=sm_30 test_random.cu -o test_random