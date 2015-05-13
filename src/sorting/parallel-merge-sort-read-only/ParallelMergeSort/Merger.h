#ifndef MERGER_H
#define MERGER_H

#include "vector_types.h"

class Merger
{
private:
int *dev_a;
int *dev_b;
int *dev_c;

	void Cleanup();
	void SwapDeviceArrays();
	dim3 GetNBlocks(int numberOfBlocks, int maxNumberOfBlocks);
	void AllocateMemoryOnDevice(int A[], int N);
	
public:
	//int CalculateNumberOfBlocks(int N);
	double PerformParallelSort(int n, int A[], int N, int I, int T, float* t_shared, float* t_nonShared);
};

#endif