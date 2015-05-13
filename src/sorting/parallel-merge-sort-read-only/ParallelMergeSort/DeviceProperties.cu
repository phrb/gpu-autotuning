#include "DeviceProperties.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <conio.h>
using std::cout;
using std::endl;

DeviceProperties::DeviceProperties()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	SharedMemoryPerBlock = deviceProp.sharedMemPerBlock;
	MaxNumberOfBlocks = deviceProp.maxGridSize[0];
	MaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	ClockFrequency = deviceProp.clockRate; // [kHz]
}

void DeviceProperties::CheckCompatability()
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) 
	{
		cout << "No CUDA-supporting device was found on the system." << endl;
		getch();
		exit(1);
	}
}