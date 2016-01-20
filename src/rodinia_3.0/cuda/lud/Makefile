CC = gcc-4.8
NVCC = nvcc -ccbin gcc-4.8

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

CFLAGS += -I../common \
					-I/usr/include/cuda \
		  -O3 \
		  -Wall \
		  $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES     := lud.c lud_cuda.c ./commonn/common.c

OBJS = ./common/common.o lud.o lud_kernel.o

.PHONY: all clean 
all : $(EXECUTABLE)

.c.o : 
	$(NVCC) $(KERNEL_DIM)  $(NVCC_FLAGS) $(DEFS) -o $@ -c $< -lcuda -lcudart -lm

%.o:	%.cu 
	$(NVCC) $(KERNEL_DIM) $(NVCC_FLAGS) $(DEFS) -o $@ -c $< -lcuda -lcudart -lm


$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@  $? -lcuda -lcudart -lm

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
