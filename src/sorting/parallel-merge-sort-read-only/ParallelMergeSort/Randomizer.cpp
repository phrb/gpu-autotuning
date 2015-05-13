#include "Randomizer.h"
#include <stdlib.h>
#include <time.h>

int* Randomizer::GenerateArray(int N, int min /* = 0 */, int max /* = 10 */)
{
	srand((unsigned int)time(NULL));
	int* A = new int[N];
	for (int i = 0; i < N; i++)
	{
		A[i] = (rand() % (max - min)) + min;
	}
	return A;
}

int* Randomizer::CreatePermutation(int N)
{
	srand((unsigned int)time(NULL));
	int* A = new int[N];
	int* Skip = new int[N];
	
	for (int i = 0; i < N; i++)
	{
		Skip[i] = i;
	}
	
	for (int i =0; i < N; i++)
	{
		int r = rand() % N;
		int p = r;
		while (Skip[r] != r)
		{
			r = Skip[r];
		}
		A[r] = i;
		Skip[r] = (r + 1) % N;
		Skip[p] = Skip[r];
	}

	delete[] Skip;
	return A;
}