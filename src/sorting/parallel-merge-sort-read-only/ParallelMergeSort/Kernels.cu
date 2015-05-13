#include "cuPrintf.cu"
#include "Common.h"

#define MEASURE_PERFORMANCE 1

namespace Kernels
{	

	__device__ float GetTimeDifference(clock_t start, clock_t stop, int clockFrequency)
	{
		return (float)(stop - start) / (clockFrequency);
	}

	/* Copies contest of Source array to Destination array.
	 * The total number of elements that will be copied are numberOfElementsToCopy.
	 * In case there are fewer threads than elements to copy, each thread will copy several elements. */
	__device__ void Copy(int Destination[], int Source[], int numberOfElementsToCopy)
	{
		//int offset = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x;
		for (int i = threadIdx.x; i < numberOfElementsToCopy; i += blockDim.x)
		{
			Destination[i] = Source[i];
		}
	}

	/* Merges arrays A & B into array C.
	 * A & B are of size szA & szB (respectively).
	 * Array C size should be at least szA + szB.
	 */
	__device__ void Merge(int A[], int B[], int C[], int szA, int szB)
	{
		int a = 0;
		int b = 0;
		int c = 0;

		while (a < szA && b < szB)
		{
			C[c++] = (A[a] < B[b]) ? A[a++] : B[b++];
		}
		while (a < szA) { C[c++] = A[a++]; }
		while (b < szB) { C[c++] = B[b++]; }	
	}

	/* Swaps contents of a & b */
	__device__ void Swap(int* a, int* b)
	{
		int* c = a;
		a = b;
		b = c;
	} 

	/* Shared memory array (of each block) 
	 * The shared memory is divided to two: 
	 * - As: first half is for source elements to be sorted.
	 * - Bs: second half is for the sorted result. (e.g. Bs = As + numberOfElementsPerBlock) */
	extern __shared__ int As[];

	/* Copies elements from source array to the shared memory.
	 * Since it is very likely that the number of elements per block is bigger than the number of 
	 * threads per block, each thread must copy more than one element to the shared memory.
	 * NOTE: numberOfElementsPerBlock should equal: Shared memory size / 2 */
	__device__ void CopyToSharedMemory(int source[], int numberOfElementsPerBlock)
	{
		int threadID = threadIdx.x;
		for (int i = threadID; i < numberOfElementsPerBlock; i += blockDim.x)
		{ 
			As[i] = source[i];
		}			
	}

	/* Copies elements from shared memory to the destination array. */
	__device__ void CopyFromSharedMemory(int destination[], int numberOfElementsPerBlock)
	{
		int* Bs = As + numberOfElementsPerBlock;
		int threadID = threadIdx.x;
		for (int i = threadID; i < numberOfElementsPerBlock; i += blockDim.x)
		{ 
			destination[i] = Bs[i];
		}	
	}

	__global__ void Sort(int source[], 
						 int destination[], 
						 int numberOfElementsPerBlock,
						 int clockFrequency,
						 float copyToSharedMemory[], // nBlocks
						 float copyFromSharedMemory[], // nBlocks
						 float mergeOperation[], // nBlocks * depth
						 float copyLayer[]/*, // nBlocks * Depth
						 int maxDepth*/) 
	{
		int blockOffset = (gridDim.x * blockIdx.y + blockIdx.x);
		source += (blockOffset * numberOfElementsPerBlock);
		destination += (blockOffset * numberOfElementsPerBlock);

/* CODE FOR: Performance monitoring for copying to shared memory. */
#if MEASURE_PERFORMANCE		
		clock_t copy_to_shared_start, copy_to_shared_stop;
		if (threadIdx.x == 0)
		{
			copy_to_shared_start = clock();
		}
#endif

/* Common code */
		__syncthreads();

		CopyToSharedMemory(source, numberOfElementsPerBlock);

		__syncthreads();
/* Common code */

#if MEASURE_PERFORMANCE
		if (threadIdx.x == 0)
		{
			copy_to_shared_stop = clock();
			float d = GetTimeDifference(copy_to_shared_start,copy_to_shared_stop, clockFrequency);			
			copyToSharedMemory[blockOffset] = d;
		}
#endif
		int* Bs = As + numberOfElementsPerBlock;

		int threadID = threadIdx.x;
		int layer = 0;
		// Assumption: numberOfElementsPerBlock is a power of 2	
		for (int d = 1; d <= (numberOfElementsPerBlock / 2); d *= 2)
		{
#if MEASURE_PERFORMANCE
			int index;
			clock_t layer_start, layer_stop;
			if (threadIdx.x == 0) layer_start = clock();
#endif
			// The number of elements per block might be bigger than number of threads per block,
			// so some threads might need to perform more than one merge		
			for (int i = threadID; (2 * i * d) < numberOfElementsPerBlock; i += blockDim.x)
			{
				// A thread no longer participate in the merge process, if its area of reponsibility overflows
				// from the shared memory.
				// However, it is kept in order to parallelize the copying process			
				int threadOffset = 2 * i * d;
				int* src1 = As+threadOffset;
				int* src2 = As+threadOffset+d;
				int* dest = Bs+threadOffset;
				Merge(src1, src2, dest, d, d);
				
			}
#if MEASURE_PERFORMANCE
			if (threadIdx.x == 0) 
			{
				layer_stop = clock();
				float d = GetTimeDifference(layer_start,layer_stop,clockFrequency);
				int totalNumberOfBlocks = gridDim.x * gridDim.y;
				index = layer * totalNumberOfBlocks + blockOffset;
				mergeOperation[index] = d;
				//cuPrintf("Layer %d sorted in %f [msec]\n", layer, diff);
			}
#endif
			layer++;
			// Let all threads finish current layer before performing the swap
			__syncthreads();

#if MEASURE_PERFORMANCE
			clock_t copy_start, copy_stop;
			if (threadIdx.x == 0) copy_start = clock();
#endif
			// Use current layer result as the base for the next layer
			
			Copy(As,Bs, numberOfElementsPerBlock);

#if MEASURE_PERFORMANCE
			if (threadIdx.x == 0) 
			{
				copy_stop = clock();
				float d = GetTimeDifference(copy_start,copy_stop,clockFrequency);
				copyLayer[index] = d;
			}
#endif

			// Let the swap finish before going on to the next layer
			__syncthreads();
			
		}	

		if (threadID == 0)
		{ // Only the 1st thread, which is the only one active in all layers, will perform the swap
			//cuPrintf("Switching array [%p] with [%p]\n", As, Bs);
			Swap(As,Bs);
		}

		__syncthreads();

#if MEASURE_PERFORMANCE		
		clock_t copy_from_shared_start, copy_from_shared_stop;
		if (threadIdx.x == 0)
		{
			copy_from_shared_start = clock();
		}
#endif

		CopyFromSharedMemory(destination, numberOfElementsPerBlock);

#if MEASURE_PERFORMANCE
		if (threadIdx.x == 0)
		{
			copy_from_shared_stop = clock();
			float d = GetTimeDifference(copy_from_shared_start,copy_from_shared_stop, clockFrequency);
			copyFromSharedMemory[blockOffset] = d;
		}
#endif
	}

	// d - size of arrays to be merged
	__global__ void Sort_NonSharedMemory(int a[], int b[], int d)
	{
		int i = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x; 
		int s = 2*d*i;
		Merge(a+s, a+s+d, b+s, d, d);
	}
}