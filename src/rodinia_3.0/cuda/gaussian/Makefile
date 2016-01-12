include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

NVCC_FLAGS = -arch sm_20


INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(NVCC_FLAGS) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
