#ifndef QPCTIMER_H
#define QPCTIMER_H
#include "Timer.h"
#include "windows.h"


class QPCTimer : public Timer
{
private:
	LARGE_INTEGER startTime;
	LARGE_INTEGER stopTime;
	LARGE_INTEGER frequency;
	double resolution;	
protected:
	void MeasureStartTime();
	void MeasureStopTime();
	double CalculateDifference();
public:
	// Creates a new instancee of the class Timer
	QPCTimer();
};

#endif