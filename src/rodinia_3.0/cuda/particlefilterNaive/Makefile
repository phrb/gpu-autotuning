#makefile

include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -ccbin gcc-4.8

INCLUDE := $(CUDA_DIR)/include

all: naive

naive: ex_particle_CUDA_naive_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(NVCC_FLAGS) -lcuda -g -lm ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
clean:
	rm particlefilter_naive
