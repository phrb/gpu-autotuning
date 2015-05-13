#ifndef MERGESORT_H
#define MERGESORT_H

class MergeSort
{
private:
	/* Summary:
	 * Copies the contents of array A into array B.
	 * Both arrays are of size N. */
	static void Copy(int A[], int B[], int N);
	/* Summary:
	 * Performs the actual sorting.
	 * A - the array to be sorted.
	 * B - The array which will contain the sorted array.
	 * C - An auxiliary array. Allocated and deallocated in Sort(). 
	 * N - Size of arrays A,B,C. */
	static void SortInternal(int A[], int B[], int C[], int N);
public:
	
	/* Summary:
	 * Merges array A and B while maintaining correct order of elements, and saves
	 * the result to array C.
	 * Array A is of size N_A.
	 * Array B is of size N_B.
	 * Array C is of size N_A + N_B;
	 * Assumption: Both arrays, A & B are sorted. */
	static void Merge(int A[], int B[], int C[], int N_A, int N_B);

	/* Summary:
	 * Sorts array A and saves the result to B using the MergeSort algorithm. 
	 * Both arrays are of size N. */
	static void Sort(int A[], int B[], int N);

	static void Sort_NR(int A[], int B[], int N, int firstLayerSize = 1);
};

#endif