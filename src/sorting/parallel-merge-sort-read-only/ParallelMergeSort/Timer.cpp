#include "Timer.h"
#include "windows.h"
#include <iostream>
using std::cerr;
using std::endl;
using std::cout;

Timer::Timer()
{
	started = false;
	paused = false;
	accumulatedTime = 0.0;
}

void Timer::Start()
{
	started = true;
	MeasureStartTime();
}

double Timer::Stop()
{
	if (started)
	{
		started = false;
		MeasureStopTime();
		return CalculateDifference();
	}
	else
	{
		cerr << "Attempted to stop a timer which hasn't started." << endl;
		return -1;
	}
}

void Timer::Print(ostringstream* str)
{
	//cout << str->str() << endl;
}