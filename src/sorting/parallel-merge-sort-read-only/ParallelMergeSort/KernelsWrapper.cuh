#ifndef KERNELS_WRAPPER
#define KERNELS_WRAPPER

#include "vector_types.h"

class KernelsWrapper
{
public:
	static void Sort(dim3 nBlocks, 
					 int T, 
					 size_t sharedSize, 
					 int* dev_A, 
					 int* dev_C, 
					 int arrayElementsPerBlock,
					 int depth);
};

#endif