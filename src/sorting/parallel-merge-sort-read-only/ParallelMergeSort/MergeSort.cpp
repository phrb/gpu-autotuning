#include "MergeSort.h"

void MergeSort::Copy(int A[], int B[], int N)
{
	for (int i = 0; i < N; i++)
	{
		B[i] = A[i];
	}
}

void MergeSort::Merge(int A[], int B[], int C[], int N_A, int N_B)
{
	int a = 0;
	int b = 0;
	int c = 0;
	
	while (a < N_A && b < N_B)
	{
		C[c++] = (A[a] < B[b]) ? A[a++] : B[b++];
	}
	while (a < N_A) { C[c++] = A[a++]; }
	while (b < N_B) { C[c++] = B[b++]; }	
}

void MergeSort::SortInternal(int A[], int B[], int C[], int N)
{
	if (N == 1)
	{
		B[0] = A[0];
	}
	else
	{
		int left = N / 2;
		int right = N - left;
		SortInternal(A,B,C,left);
		SortInternal(A+left,B+left,C,right);
		Merge(B, B+left, C, left, right);
		Copy(C,B,N);		
	}
}

void MergeSort::Sort(int A[], int B[], int N)
{
	int* C = new int[N];
	SortInternal(A,B,C,N);
	delete[] C;
}

void MergeSort::Sort_NR( int A[], int B[], int N, int firstLayerSize)
{
	int* C = new int[N];
	Copy(A,C,N);
	// base is the size of the sub-array we currently sort
	for (int len = firstLayerSize; len <= N/2; len *= 2)
	{
		for (int base = 0; base < N; base += (2*len))
		{
			Merge(C+base,C+base+len,B+base,len, len);
			Copy(B+base,C+base,len*2);
		}
	}
	delete[] C;
}

