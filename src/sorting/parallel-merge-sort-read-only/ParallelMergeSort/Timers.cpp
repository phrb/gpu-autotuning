#include "Timers.h"
#include "Common.h"
#include <cuda_runtime.h>
#include "float.h"

#include <iostream>
#include <fstream>

using std::cout;
using std::endl;
using std::ofstream;

Timers* Timers::Allocate(bool isOnHost, int NumberOfBlocks, int Depth)
{
	if (isOnHost)
	{
		Timers* timers = new Timers();
		timers->CopyToSharedMemory = new float[NumberOfBlocks];
		timers->CopyFromSharedMemory = new float[NumberOfBlocks];
		timers->MergeOperation = new float[NumberOfBlocks * Depth];
		timers->CopyLayer = new float[NumberOfBlocks * Depth];
		return timers;
	}
	else
	{
		Timers* timers = new Timers();
		cudaError_t errorCode;
		errorCode = cudaMalloc<float>(&timers->CopyToSharedMemory, sizeof(float) * NumberOfBlocks);
		ASSERT(errorCode);

		errorCode = cudaMalloc<float>(&timers->CopyFromSharedMemory, sizeof(float) * NumberOfBlocks);
		ASSERT(errorCode);

		errorCode = cudaMalloc<float>(&timers->MergeOperation, sizeof(float) * NumberOfBlocks * Depth);
		ASSERT(errorCode);

		errorCode = cudaMalloc<float>(&timers->CopyLayer, sizeof(float) * NumberOfBlocks * Depth);
		ASSERT(errorCode);
		return timers;
	}
}

void Timers::Destroy(Timers* timers, bool OnHost)
{
	if (OnHost)
	{
		delete[] timers->CopyFromSharedMemory;
		delete[] timers->CopyToSharedMemory;		
		delete[] timers->MergeOperation;
		delete[] timers->CopyLayer;
		delete timers;
	}
	else
	{
		cudaFree(timers->CopyFromSharedMemory);
		cudaFree(timers->CopyToSharedMemory);		
		cudaFree(timers->MergeOperation);
		cudaFree(timers->CopyLayer);
		delete timers;
	}
}

Timers* Timers::CopyToHost(Timers* deviceTimers, int numberOfBlocks, int depth)
{
	Timers* hostTimers = Timers::Allocate(true, numberOfBlocks, depth);
	cudaError_t errorCode;
	errorCode = cudaMemcpy(hostTimers->CopyFromSharedMemory, 
						   deviceTimers->CopyFromSharedMemory, 
						   sizeof(float) * numberOfBlocks, 
						   cudaMemcpyDeviceToHost);
	ASSERT(errorCode);

	errorCode = cudaMemcpy(hostTimers->CopyToSharedMemory, 
						   deviceTimers->CopyFromSharedMemory,
						   sizeof(float) * numberOfBlocks, 
						   cudaMemcpyDeviceToHost);
	ASSERT(errorCode);

	errorCode = cudaMemcpy(hostTimers->CopyLayer, 
						   deviceTimers->CopyLayer,
						   sizeof(float) * numberOfBlocks * depth, 
						   cudaMemcpyDeviceToHost);
	ASSERT(errorCode);

	errorCode = cudaMemcpy(hostTimers->MergeOperation, 
						   deviceTimers->MergeOperation,
						   sizeof(float) * numberOfBlocks * depth, 
						   cudaMemcpyDeviceToHost);
	ASSERT(errorCode);
	return hostTimers;
}

float Timers::Average(float d[], int N)
{
	float t = 0;	

	for (int i = 0; i < N; i++)
	{
		t *= i;
		t += d[i];
		t /= (i+1);
	}

	return t;
}

float Timers::Max(float d[], int N)
{
	float m = -FLT_MAX;
	for (int i = 0; i < N; i++)
	{
		m = (d[i] > m) ? d[i] : m;
	}
	return m;
}

float Timers::Min(float d[], int N)
{
	float m = FLT_MAX;
	for (int i = 0; i < N; i++)
	{
		m = (d[i] < m) ? d[i] : m;
	}
	return m;	
}

void Timers::PrintSummary(Timers* timers, int numberOfBlocks, int depth)
{
	ofstream copyTo("D:\\CopyTo.txt", std::ios::app);
	ofstream copyFrom("D:\\CopyFrom.txt", std::ios::app);
	ofstream layerMerge("D:\\LayerMerge.txt", std::ios::app);
	ofstream layerCopy("D:\\LayerCopy.txt", std::ios::app);

	float copyFromSharedAverage = Timers::Average(timers->CopyFromSharedMemory, numberOfBlocks);
	float copyFromSharedMin = Timers::Min(timers->CopyFromSharedMemory, numberOfBlocks);
	float copyFromSharedMax = Timers::Max(timers->CopyFromSharedMemory, numberOfBlocks);
	float copyToSharedAverage = Timers::Average(timers->CopyToSharedMemory, numberOfBlocks);
	float copyToSharedMin = Timers::Min(timers->CopyToSharedMemory, numberOfBlocks);
	float copyToSharedMax = Timers::Max(timers->CopyToSharedMemory, numberOfBlocks);

	//cout << "Copy to Shared Memory: [" << copyToSharedMin << ", " << copyToSharedMax << "]. Avg: " << copyToSharedAverage << endl;
	//cout << "Copy from Shared Memory: [" << copyFromSharedMin << ", " << copyFromSharedMax << "]. Avg: " << copyFromSharedAverage << endl;
	copyTo << "CopyToSharedMemory" << '\t' << copyToSharedMin << '\t' << copyToSharedMax << '\t' << copyToSharedAverage << endl;
	copyFrom << "CopyFromSharedMemory" << '\t' << copyFromSharedMin << '\t' << copyFromSharedMax << '\t' << copyFromSharedAverage << endl;
	for (int i = 0; i < depth; i++)
	{
		float* mergePtr = timers->MergeOperation + numberOfBlocks * i;
		float mergeAverage = Timers::Average(mergePtr, numberOfBlocks);
		float mergeMin = Timers::Min(mergePtr, numberOfBlocks);
		float mergeMax = Timers::Max(mergePtr, numberOfBlocks);

		//cout << "Layer " << i << " Merge: [" << mergeMin << ", " << mergeMax << "]. Avg: " << mergeAverage << endl;
		layerMerge << "LayerMerge" << '\t' << i << '\t' << mergeMin << '\t' << mergeMax << '\t' << mergeAverage << endl;
	}

	for (int i = 0; i < depth; i++)
	{
		float* copyPtr = timers->CopyLayer + numberOfBlocks * i;
		float copyAverage = Timers::Average(copyPtr, numberOfBlocks);
		float copyMin = Timers::Min(copyPtr, numberOfBlocks);
		float copyMax = Timers::Max(copyPtr, numberOfBlocks);

		//cout << "Layer " << i << " Copy: [" << copyMin << ", " << copyMax << "]. Avg: " << copyAverage << endl;
		layerCopy << "LayerCopy" << '\t' << i << '\t' << copyMin << '\t' << copyMax << '\t' << copyAverage << endl;
	}

	copyTo.close();
	copyFrom.close();
	layerMerge.close();
	layerCopy.close();
}