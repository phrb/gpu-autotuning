#ifndef TIMER_H
#define TIMER_H

// 0 - ClockTimer, 1 - QPCTimer, 2 - CudaTimer
#define TIMER_TYPE 1

#include <string>
#include <sstream>
using std::ostringstream;

class Timer
{
protected:
	bool started;
	bool paused;
	double accumulatedTime;
	virtual void MeasureStartTime() = 0;
	virtual void MeasureStopTime() = 0;
	virtual double CalculateDifference() = 0;
public:
	// Starts a timer
	Timer();
	void Start();
	// Stops the timer and returns the time in msecs
	double Stop();
	static void Print(ostringstream* str);
};

#endif