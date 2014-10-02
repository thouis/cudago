all: board19 board13 board9


board19: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=500 -DBOARD_SIZE=19 -G -g -arch=sm_30 board.cu -o board19

board13: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=2500 -DBOARD_SIZE=13 -O3 -arch=sm_30 board.cu -o board13

board9: board.cu
	/usr/local/cuda/bin/nvcc -DPLAYOUT_COUNT=2500 -DBOARD_SIZE=9 -O3 -arch=sm_30 board.cu -o board9
