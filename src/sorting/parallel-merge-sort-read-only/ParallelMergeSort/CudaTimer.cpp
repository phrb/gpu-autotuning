#include "CudaTimer.h"

CudaTimer::CudaTimer()
{
	cudaEventCreate(&startEvt);
	cudaEventCreate(&endEvt);
}

void CudaTimer::MeasureStartTime()
{
	cudaEventRecord(startEvt);
	cudaEventSynchronize(startEvt);
}

void CudaTimer::MeasureStopTime()
{
	cudaEventRecord(endEvt);
	cudaEventSynchronize(endEvt);
}

double CudaTimer::CalculateDifference()
{
	float totalTime;
	cudaEventElapsedTime(&totalTime, startEvt, endEvt);
	return (double)totalTime;
}