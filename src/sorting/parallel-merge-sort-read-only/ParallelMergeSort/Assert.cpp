#include "Assert.h"
#include <iostream>
#include <conio.h>
#include "cuda_runtime_api.h"
using std::cout;
using std::endl;

bool Assert::IsSorted(int A[], int N)
{
	for (int i = 0; i < N - 1; i++)
	{
		if (A[i] > A[i+1]) return false;
	}
	return true;
}

bool Assert::CudaActionSuccessful( cudaError_t e , const char* file, int line, void(*Action)(void))
{
	if (e != cudaSuccess)
	{
		cout << "CUDA returned error: '" << cudaGetErrorString(e) << "'" << endl;
		cout << "Error in " << file << ":" << line << endl;
		getch();
		if (Action != NULL) Action();
		return false;
	}
	else return true;
}
