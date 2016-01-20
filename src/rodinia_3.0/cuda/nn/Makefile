include ../../common/make.config

LOCAL_CC = gcc-4.8 -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc 

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) $(NVCC_FLAGS) -cuda nn_cuda.cu -lcuda -lcudart -lm
	$(CC) $(NVCC_FLAGS) -o nn nn_cuda.cu -lcuda -lcudart -lm

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
