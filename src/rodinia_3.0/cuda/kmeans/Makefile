include ../../common/make.config

# C compiler
CC = gcc-4.8
CC_FLAGS = -g -fopenmp  

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc -ccbin gcc-4.8

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

%.o: %.[c]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu -lcuda -lcudart -lm

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
