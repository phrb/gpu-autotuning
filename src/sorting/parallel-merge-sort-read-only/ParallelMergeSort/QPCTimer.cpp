#include "QPCTimer.h"
#include "winbase.h"
#include <iostream>
using std::cerr;
using std::endl;

QPCTimer::QPCTimer()
{
	startTime.QuadPart = 0;
	stopTime.QuadPart = 0;
	QueryPerformanceFrequency(&frequency);
	resolution = (double)frequency.QuadPart;
}

void QPCTimer::MeasureStartTime()
{	
	QueryPerformanceCounter(&startTime);	
}

void QPCTimer::MeasureStopTime()
{	
	QueryPerformanceCounter(&stopTime);	
}

double QPCTimer::CalculateDifference()
{
	LONGLONG diff = stopTime.QuadPart - startTime.QuadPart;	
	return diff / resolution * 1000;
}

