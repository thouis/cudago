board: board.cu
	/usr/local/cuda/bin/nvcc -O3 -arch=sm_30 board.cu -o board
