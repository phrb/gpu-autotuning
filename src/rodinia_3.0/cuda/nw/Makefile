include ../../common/make.config

CC := nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

release: $(SRC)
	$(CC) ${KERNEL_DIM} $(SRC) $(NVCC_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) $(NVCC_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) $(NVCC_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) $(NVCC_FLAGS) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo
