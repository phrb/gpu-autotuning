#ifndef TIMERS_H
#define TIMERS_H

class Timers
{
private:
	static float Average(float d[], int N);
	static float Max(float d[], int N);
	static float Min(float d[], int N);
public:
	float* CopyToSharedMemory;
	float* CopyFromSharedMemory;
	float* MergeOperation;
	float* CopyLayer;

	static Timers* Allocate(bool onHost, int numberOfBlocks, int depth);
	static Timers* CopyToHost(Timers* deviceTimers, int numberOfBlocks, int depth);
	static void Destroy(Timers* timers, bool onHost);
	// Can only be performed on host
	static void PrintSummary(Timers* timers, int numberOfBlocks, int depth);
};


#endif