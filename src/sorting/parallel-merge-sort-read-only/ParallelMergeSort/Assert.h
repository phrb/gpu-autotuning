#ifndef ASSERTER_H
#define ASSERTER_H
#include "driver_types.h"

class Assert
{
public:
	/* Summary:
	 * Verifies that the array A is sorted.
	 * Array A is of size N.
	 * Returns "true" if array is sorted, otherwise, "false". */
	static bool IsSorted(int A[], int N);
	/* Summary:
	 * Verifies that the action resulting in the given cudaError_t was successful.
	 * If cudaError_t != cudaSuccess, the application prints the error code and EXITS. */
	static bool CudaActionSuccessful(cudaError_t, const char*, int, void(*Action)(void));
};

#endif