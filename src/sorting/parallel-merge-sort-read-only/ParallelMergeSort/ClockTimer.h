#ifndef CLOCK_TIMER_H
#define CLOCK_TIMER_H
#include <ctime>
#include "Timer.h"

class ClockTimer : public Timer
{
private:
	clock_t startTime;
	clock_t stopTime;
protected:
	void MeasureStartTime();
	void MeasureStopTime();
	double CalculateDifference();
public:
	// Creates a new instancee of the class Timer
	ClockTimer();
};

#endif