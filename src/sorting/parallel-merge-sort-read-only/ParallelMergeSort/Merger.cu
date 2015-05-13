#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include "MergeSort.h"
#include "Randomizer.h"
#include "Assert.h"
#include "QPCTimer.h"
#include "ClockTimer.h"
#include "Kernels.cu"
#include "cuPrintf.cuh"
#include "Common.h"
#include "DeviceProperties.cuh"

#define ASSERT(errorCode) \
	Assert::CudaActionSuccessful(errorCode, __FILE__, __LINE__, NULL)

using namespace std;

//TODO: Make these not global (currently global for Cleanup())
int *dev_a;
int *dev_b;
int *dev_c;

void Cleanup()
{
	// Free allocated device memory
	if (dev_a)
	{
		cudaFree(dev_a);
		dev_a = NULL;
	}
	if (dev_c)
	{
		cudaFree(dev_c);
		dev_c = NULL;
	}	
}

void SwapDeviceArrays()
{
	dev_b = dev_a;
	dev_a = dev_c;
	dev_c = dev_b;
}

dim3 GetNBlocks(int numberOfBlocks, int maxNumberOfBlocks)
{
	if (numberOfBlocks > maxNumberOfBlocks)
	{
		return dim3(maxNumberOfBlocks, numberOfBlocks / maxNumberOfBlocks);
	}
	else
	{
		return dim3(numberOfBlocks, 1);
	}
}

void AllocateMemoryOnDevice(int A[], int N)
{
	cudaError_t errorCode;
	errorCode = cudaMalloc<int>(&dev_a, sizeof(int) * N);
	ASSERT(errorCode);

	errorCode = cudaMalloc<int>(&dev_c, sizeof(int) * N);
	ASSERT(errorCode);

	errorCode = cudaMemcpy(dev_a, A, sizeof(int)*N, cudaMemcpyHostToDevice);
	ASSERT(errorCode);
}

float PerformParallelSort_WithSharedMemory(int n, int N, int I, int T)
{
	float totalTime_p;	
	float totalSortingTime;
	float totalCopyingTime;
	cudaEvent_t startEvent_p, endEvent_p;
	cudaEvent_t startCopy, endCopy;
	cudaEvent_t startSort, endSort;
	cudaError_t errorCode;
// Creating events for measuring performance
	cudaEventCreate(&startEvent_p);
	cudaEventCreate(&endEvent_p);
	cudaEventCreate(&startCopy);
	cudaEventCreate(&endCopy);
	cudaEventCreate(&startSort);
	cudaEventCreate(&endSort);
	
	cudaEventRecord(startEvent_p);

	int* C = new int[N];
	DeviceProperties deviceProps;

	size_t sharedSize = min(2*N*sizeof(int), deviceProps.SharedMemoryPerBlock / 2);
	int arrayElementsPerBlock = sharedSize / sizeof(int);
	int numberOfBlocks = (N / (arrayElementsPerBlock/2)) + ((N % (arrayElementsPerBlock/2) > 0) ? 1 : 0);
	
	dim3 nBlocks = GetNBlocks(numberOfBlocks, deviceProps.MaxNumberOfBlocks);
	// NumberOfElementsPerBlock = Shared memory size / 2
	// Sort<<<NumberOfBlocks, NumberOfThreadsPerBlock, SizeOfSharedMemory>>>
	//		(Source,Destination,NumberOfElementsPerBlock)

	cudaEventRecord(startSort);
	for (int i = 0; i < I; i++)
	{
		Sort<<<nBlocks,T,sharedSize>>>(dev_a,dev_c,arrayElementsPerBlock / 2);
		cudaThreadSynchronize();
		errorCode = cudaGetLastError();
		
		if (!ASSERT(errorCode)) break;

		if (numberOfBlocks > 1) SwapDeviceArrays();
		
		for (int d = arrayElementsPerBlock / 2; d <= (N/2); d *= 2)
		{			
			int nThreads = min(T, N / (2*d));
			int numberOfBlocks = N / (2*d*nThreads);
			nBlocks = GetNBlocks(numberOfBlocks, deviceProps.MaxNumberOfBlocks);
			Sort_NonSharedMemory<<<nBlocks,nThreads>>>(dev_a, dev_c, d);
			cudaThreadSynchronize();
			errorCode = cudaGetLastError();
			ASSERT(errorCode);
			
			// Swap contents of dev_c & dev_a
			SwapDeviceArrays();
		}
	}

	if (numberOfBlocks > 1) SwapDeviceArrays();

	cudaEventRecord(endSort);
	cudaEventSynchronize(endSort);
	cudaEventElapsedTime(&totalSortingTime, startSort, endSort);
	
	cudaEventRecord(startCopy);
	errorCode = cudaMemcpy(C, dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaEventRecord(endCopy);
	cudaEventSynchronize(endCopy);
	cudaEventElapsedTime(&totalCopyingTime, startCopy, endCopy);

	ASSERT(errorCode);

	cudaEventRecord(endEvent_p);
	cudaEventSynchronize(endEvent_p);
	cudaEventElapsedTime(&totalTime_p, startEvent_p, endEvent_p);

	delete[] C;
	return totalTime_p / I;
}


float PerformParallelSort(int n, int A[], int N, int I, int T, float* t_shared, float* t_nonShared)
{
	float totalTime_p;	
	cudaEvent_t startEvent_p, endEvent_p;
	cudaEventCreate(&startEvent_p);
	cudaEventCreate(&endEvent_p);

	cudaEventRecord(startEvent_p);
	
	AllocateMemoryOnDevice(A,N);	

	cudaEventRecord(endEvent_p);
	cudaEventSynchronize(endEvent_p);
	cudaEventElapsedTime(&totalTime_p, startEvent_p, endEvent_p);
	
	totalTime_p = PerformParallelSort_WithSharedMemory(n, N, I, T);
		
	Cleanup();
	return totalTime_p;
}

int main(int argc, char* argv[])
{
	int N; // Number of elements in the array
	int I; // Number of iterations to perform
	int T; // Number of threads per block	
	
//	Common::ReadParameters(&N, &I, &T);
	N = 2048;
	I = 1;
	T = 32;

	Timer* timer = new QPCTimer();	
	timer->Start();
	for (int i = 0; i < 10; i++)
	{
		Sleep(1000);
		cout << i+1 << " Mississippi" << endl;
	}
	double t2 = timer->Stop();
	cout << "Slept for " << t2 << "[msec] (according to QueryPerformanceMonitor)" << endl;	

	Timer* timer1 = new ClockTimer();	
	timer1->Start();
	for (int i = 0; i < 10; i++)
	{
		Sleep(1000);
		cout << i+1 << " Mississippi" << endl;
	}
	double t1 = timer1->Stop();
	cout << "Slept for " << t1 << "[msec] (according to clock())" << endl;	

	int* A; 
	A = Randomizer::GenerateArray(N,0,1000000);
	float s_t = Common::PerformSequentialSort(A, N, I);
    cudaPrintfInit();	
	
	float p_t1 = PerformParallelSort(11,A,N,I,T, NULL, NULL);
	

	cudaPrintfDisplay(stdout, true);	
	cudaPrintfEnd();
	
	cout << "Sequential: " << s_t << endl;
	cout << "Parallel with Shared Memory: " << p_t1 << " (" << p_t1 / s_t * 100 << "%)" << endl;
	
	getch();
}