#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>
#include "Timer.h"

class CudaTimer : public Timer
{
private:
	cudaEvent_t startEvt;
	cudaEvent_t endEvt;
protected:
	void MeasureStartTime();
	void MeasureStopTime();
	double CalculateDifference();
public:
	// Creates a new instancee of the class Timer
	CudaTimer();
};


#endif