#ifndef COMMON_H
#define COMMON_H

#include "Assert.h"

#define ASSERT(errorCode) \
	Assert::CudaActionSuccessful(errorCode, __FILE__, __LINE__, NULL)

class Common
{
private:
	
	static void CheckCompatibility();
public:
	/* Reads parameters from the user.
	 * D - Depth (2^D = N)
	 * N - Number of elements in the array.
	 * I - Number of iterations to perform.
	 * T - Number of threads per block. */
	static void ReadParameters(int* D, int* N, int* I, int* T);

	/* Performs a I iterations of sequential sort on array A of size N.
	 * Return the total time (in msec) of the entire operation. 
	 * The array is sorted, and checked for sorting. */
	static double PerformSequentialSort(int A[], int N, int I);

	/* Prints array A of size N */
	static void PrintArray(int A[], int N);

	static void PrintArrayToFile(const char* filename, int A[], int N);
};

#endif