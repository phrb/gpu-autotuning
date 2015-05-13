#ifndef RANDOMIZER_H
#define RANDOMIZER_H

class Randomizer
{
public:
	/* Summary:
	 * Generates an array of N elements, ranging from min to max.
	 * The array is dynamically allocated, and should be released by the user of this array. */
	static int* GenerateArray(int N, int min = 0, int max = 9);

	static int* CreatePermutation(int N);
};

#endif