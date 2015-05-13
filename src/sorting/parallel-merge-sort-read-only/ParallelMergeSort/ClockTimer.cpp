#include "ClockTimer.h"
#include <iostream>
using std::cerr;
using std::endl;

ClockTimer::ClockTimer()
{
	startTime = 0;
	stopTime = 0;
}

void ClockTimer::MeasureStartTime()
{
	startTime = clock();
}

void ClockTimer::MeasureStopTime()
{
	stopTime = clock();
}

double ClockTimer::CalculateDifference()
{
	return (stopTime - startTime) / CLOCKS_PER_SEC * 1000;
}

