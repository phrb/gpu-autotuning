#include "Common.h"
#include <iostream>
#include <fstream>
#include "DeviceProperties.cuh"
#include <cmath>
#include <conio.h>
#include "Timer.h"
#include "QPCTimer.h"
#include "MergeSort.h"
#include "Assert.h"
using namespace std;

void Common::ReadParameters(int* D, int* N, int* I, int* T)
{
	DeviceProperties::CheckCompatability();
	// Read number of elements in the array	
	*D = 0;
	while (*D < 1)
	{
		cout << "Enter number of elements (2^X). X = ";
		cin >> *D;
	}	
	*N = (int)pow(2.0, *D);

	// Read number of iterations
	*I = 0;
	while (*I < 1 || *I > 1000)
	{
		cout << "Enter number of iterations [1-1000]: ";
		cin >> *I;
	}

	DeviceProperties devProps;

	*T = 0;
	while (*T < 1 || *T > devProps.MaxThreadsPerBlock)
	{
		cout << "Enter number of threads per block [1-" << devProps.MaxThreadsPerBlock << "]: ";
		cin >> *T;
	}
}


double Common::PerformSequentialSort( int A[], int N, int I )
{
	int* B = new int[N];
	int* C = new int[N];

	Timer* sequentialTimer = new QPCTimer();
	sequentialTimer->Start();

	for (int i = 0; i < I; i++)
		MergeSort::Sort_NR(A,B,N);

	double totalTime_s = sequentialTimer->Stop();
	
	return totalTime_s / I;
}

void Common::PrintArrayToFile(const char* filename, int A[], int N)
{
	ofstream out;
	out.open(filename);
	if (out.is_open())
	{
		for (int i = 0; i < N; i++)
		{
			out << "[" << i << "] = " << A[i] << endl;
		}
		out.close();		
	}
}

void Common::PrintArray( int* b, int n ) 
{
	for (int i = 0; i < n; i++) 
	{
		cout << b[i] << " ";
	}
	cout << endl;
}
