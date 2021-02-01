include ../../common/make.config

CC := nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(NVCC_FLAGS) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

clean:
	rm gaussian
