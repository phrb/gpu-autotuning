#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include "MergeSort.h"
#include "Randomizer.h"
#include "KernelsWrapper.cuh"
#include "cuPrintf.cuh"
#include "Common.h"
#include "DeviceProperties.cuh"
#include "Merger.h"
#include "Timer.h"
#include <cmath>

#if (TIMER_TYPE == 0)
#include "ClockTimer.h"
typedef ClockTimer Timer_t;
#endif

#if (TIMER_TYPE == 1)
#include "QPCTimer.h"
typedef QPCTimer Timer_t;
#endif

#if (TIMER_TYPE == 2)
#include "CudaTimer.h"
typedef CudaTimer Timer_t;
#endif
using namespace std;

//TODO: Make these not global (currently global for Cleanup())
int *dev_a;
int *dev_b;
int *dev_c;

void Merger::Cleanup()
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

void Merger::SwapDeviceArrays()
{
	dev_b = dev_a;
	dev_a = dev_c;
	dev_c = dev_b;
}

dim3 Merger::GetNBlocks(int numberOfBlocks, int maxNumberOfBlocks)
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

void Merger::AllocateMemoryOnDevice(int A[], int N)
{
	Timer* timer = new Timer_t();
	timer->Start();
	cudaError_t errorCode;
	errorCode = cudaMalloc<int>(&dev_a, sizeof(int) * N);
	ASSERT(errorCode);

	errorCode = cudaMalloc<int>(&dev_c, sizeof(int) * N);
	ASSERT(errorCode);

	errorCode = cudaMemcpy(dev_a, A, sizeof(int)*N, cudaMemcpyHostToDevice);
	ASSERT(errorCode);
	double time = timer->Stop();
	ostringstream str;
	str << "Memory Allocation (global): " << time << " [msec]";
	timer->Print(&str);
}

double Merger::PerformParallelSort(int D, int A[], int N, int I, int T, float* t_shared, float* t_nonShared)
{
	AllocateMemoryOnDevice(A,N);	
	ofstream out("D:\\Stats.txt", std::ios::app);

	cudaError_t errorCode;
	int* C = new int[N];
	DeviceProperties deviceProps;

	Timer* timer = new Timer_t();
	timer->Start();
	
	size_t sharedSize = min(2*N*sizeof(int), deviceProps.SharedMemoryPerBlock / 2);
	int arrayElementsPerBlock = sharedSize / sizeof(int);
	int numberOfBlocks = (N / (arrayElementsPerBlock/2)) + ((N % (arrayElementsPerBlock/2) > 0) ? 1 : 0);
	
	dim3 nBlocks = GetNBlocks(numberOfBlocks, deviceProps.MaxNumberOfBlocks);
	
	// NumberOfElementsPerBlock = Shared memory size / 2
	// Sort<<<NumberOfBlocks, NumberOfThreadsPerBlock, SizeOfSharedMemory>>>
	//		(Source,Destination,NumberOfElementsPerBlock)
	
	for (int i = 0; i < I; i++)
	{
		Timer* deviceSort = new Timer_t();
		deviceSort->Start();
		int d = (int)(log((double)arrayElementsPerBlock / 2) / log(2.0));
		KernelsWrapper::Sort(nBlocks,T,sharedSize,dev_a,dev_c, arrayElementsPerBlock, d);
		cudaThreadSynchronize();
		errorCode = cudaGetLastError();
		
		if (!ASSERT(errorCode)) break;

		if (numberOfBlocks > 1) SwapDeviceArrays();
		double deviceTime = deviceSort->Stop();
		
		
		Timer* copyTimer = new Timer_t();
		copyTimer->Start();
		int* A = new int[N];
		errorCode = cudaMemcpy(A, dev_a, sizeof(int)*N, cudaMemcpyDeviceToHost);
		ASSERT(errorCode);
		double copyTime = copyTimer->Stop();

		Timer* cpuTimer = new Timer_t();
		cpuTimer->Start();
		MergeSort::Sort_NR(A,C,N,arrayElementsPerBlock / 2);
		double cpuTime = cpuTimer->Stop();

		//cout << endl << "*** Iteration " << i << " completed. ***" << endl;
		ostringstream devicestr, copystr, cpustr;
		devicestr << "Device time: " << deviceTime << " [msec]";
		copystr << "Copy to host: " << copyTime << " [msec]";
		cpustr << "CPU time: " << cpuTime << " [msec]";
		
		deviceSort->Print(&devicestr);
		copyTimer->Print(&copystr);
		cpuTimer->Print(&cpustr);

		out << "DeviceTime" << '\t' << deviceTime << endl;
		out << "CopyToHost" << '\t' << copyTime << endl;
		out << "CPUTime" << '\t' << cpuTime << endl;
		out.close();
		
		delete deviceSort;
		delete copyTimer;
		delete cpuTimer;
	}

	delete[] C;
	double totalTime_p = timer->Stop();
	ostringstream str;
	str << "Parallel sort - Total: " << totalTime_p << " [msec]";
	timer->Print(&str);
	
	Timer* cleanupTimer = new Timer_t();
	cleanupTimer->Start();
	Cleanup();
	double cleanupTime = cleanupTimer->Stop();
	ostringstream str2;
	str << "Clean up: " << cleanupTime << " [msec]";
	cleanupTimer->Print(&str2);
	return totalTime_p;
}

int main(int argc, char* argv[])
{
	int D; // Depth (2^D = N)
	int N; // Number of elements in the array
	int I; // Number of iterations to perform
	int T; // Number of threads per block	
	
	Common::ReadParameters(&D, &N, &I, &T);	

	cout << "Preparing array... ";
	int* A; 
	A = Randomizer::GenerateArray(N,0,1000000);
	cout << "Done!" << endl;
	double s_t = Common::PerformSequentialSort(A, N, I);
	
	Merger merger;
	double p_t = merger.PerformParallelSort(D,A,N,I,T, NULL, NULL);
	
	cout << "Sequential: " << s_t << endl;
	cout << "Parallel: " << p_t << endl;
}