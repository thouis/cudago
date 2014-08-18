board: board.cu
	/usr/local/cuda/bin/nvcc -arch=sm_30 board.cu -o board
