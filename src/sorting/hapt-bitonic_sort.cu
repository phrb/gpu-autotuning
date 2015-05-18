/**
 *   Bitonic Sort on GPU (using CUDA)
 *
 *   Adapted from nVidia SDK Bitonic Sort:
 *
 *     http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 *
 * Maximo, Andre -- Mar, 2009
 *
 **/

#include <cuda.h>
#include <cutil.h>

uint pofElements, szPackedArrayBitonic;

uint_64 *d_packedArrayBitonic;

/// THESE VALUES SHOULD DEPEND ON THE GPU ON WHICH WE'RE RUNNING
#define SBLOCK_SIZE_R 512
#define SBLOCK_SIZE2_R 1024
#define LOG_SBLOCK_SIZE_R 9

uint3 dimBlockBitonic = { 1, 1, 1 };
uint3 dimGridBitonic = { 1, 1, 1 };

__shared__ uint_64 shared[ SBLOCK_SIZE2_R ];

/// Bitonic Merge Block on GPU (called from CPU)

__global__
static void bitonicMergeBlock( uint_64* values, int k, int j, int lj ) {

	const int tid = threadIdx.x;
	const int bid = (blockIdx.x<<(1+LOG_SBLOCK_SIZE_R));

	shared[tid] = values[bid + tid];
	shared[tid+SBLOCK_SIZE_R] = values[bid + tid + SBLOCK_SIZE_R];
	
	__syncthreads();
        
	for (; j>0; --lj, j>>=1) {

		int sB = (tid>>lj);
		int sid = (tid&(j-1));
	
		int a = (sB<<(lj+1)) + sid;
		int b = a+j;

		if( ((a+bid) & k) == 0 ) {

			if( shared[a] > shared[b] ) {

				uint_64 tmp = shared[a];
				shared[a] = shared[b];
				shared[b] = tmp;

			}

		} else {

			if( shared[a] < shared[b] ) {

				uint_64 tmp = shared[a];
				shared[a] = shared[b];
				shared[b] = tmp;

			}

		}

		__syncthreads();

	}

	/// Write back result

	values[bid + tid] = shared[tid];
	values[bid + tid + SBLOCK_SIZE_R] = shared[tid+SBLOCK_SIZE_R];

}

/// Bitonic Merge on GPU (called from CPU)

__global__
static void bitonicMerge( uint_64* values, int k, int j, int lj ) {

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	int sS = (j >> LOG_SBLOCK_SIZE_R);
	int sB = (bid >> (lj-LOG_SBLOCK_SIZE_R));
	int sid = (bid & (sS-1));

	int a = (sB<<(lj+1)) + (sid<<LOG_SBLOCK_SIZE_R) + tid;
	int b = a+j;

	if( ((a & k) == 0) ) {

		if( (values[a] > values[b]) ) {

			uint_64 tmp = values[a];
			values[a] = values[b];
			values[b] = tmp;

		}

	} else {

		if( (values[a] < values[b]) ) {

			uint_64 tmp = values[a];
			values[a] = values[b];
			values[b] = tmp;

		}

	}

}

/// Initialize Bitonic

__host__
void initBitonic( uint numElements ) {

	/// Bitonic Sort
	pofElements = (uint) pow(2, ceil( log2f( numElements ) ) );

	dimGridBitonic.x = (uint) ceil( ((float)pofElements) / SBLOCK_SIZE2_R );

	dimBlockBitonic.x = SBLOCK_SIZE_R;

	szPackedArrayBitonic = pofElements * sizeof(uint_64);

	if( cudaMalloc( (void**) &d_packedArrayBitonic, szPackedArrayBitonic) != cudaSuccess ) { printf("!error! Allocating memory for bitonic array!\n"); return; }

	if( cudaMemset( (void*) d_packedArrayBitonic, 0xFF, szPackedArrayBitonic) != cudaSuccess ) { printf("!error! Setting memory in bitonic array!\n"); return; }

}

/// Bitonic Sort

__host__
void bitonicSort( void ) {

	for (int k = 2, lk = 1; k <= pofElements; ++lk, k <<= 1) {

		int j = (k>>1);

		int lj = lk-1;

		for (; j > SBLOCK_SIZE_R; --lj, j >>= 1) {

			bitonicMerge<<< dimGridBitonic, dimBlockBitonic >>>( d_packedArrayBitonic, k, j, lj );

			cudaThreadSynchronize();

		}

		bitonicMergeBlock<<< dimGridBitonic, dimBlockBitonic >>>( d_packedArrayBitonic, k, j, lj );

		cudaThreadSynchronize();

	}

}

/// Clean Bitonic

__host__
void cleanBitonic( void ) {

 	CUDA_SAFE_CALL( cudaFree(d_packedArrayBitonic) );

}
