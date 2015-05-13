#include "KernelsWrapper.cuh"
#include "Kernels.cu"
#include "DeviceProperties.cuh"
#include "Timers.h"
#include <iostream>
using std::cout;
using std::endl;

void KernelsWrapper::Sort(dim3 nBlocks, int T, size_t sharedSize, int* dev_A, int* dev_C, int arrayElementsPerBlock, int depth)
{		
	DeviceProperties deviceProps;
	int clockFrequency = deviceProps.ClockFrequency;
	int numberOfBlocks = nBlocks.x * nBlocks.y;
	Timers* deviceTimers = Timers::Allocate(false, numberOfBlocks, depth);
	
	Kernels::Sort<<<nBlocks,T,sharedSize>>>(dev_A,
											dev_C,
											arrayElementsPerBlock / 2, 
											clockFrequency,
											deviceTimers->CopyToSharedMemory,
											deviceTimers->CopyFromSharedMemory,
											deviceTimers->MergeOperation,
											deviceTimers->CopyLayer);

	Timers* hostTimers = Timers::CopyToHost(deviceTimers, numberOfBlocks, depth);
	Timers::PrintSummary(hostTimers, numberOfBlocks, depth);
	Timers::Destroy(deviceTimers, false);
	Timers::Destroy(hostTimers, true);
}