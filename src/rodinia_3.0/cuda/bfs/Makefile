include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -ccbin gcc-4.8

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) 

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) $(NVCC_FLAGS) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
