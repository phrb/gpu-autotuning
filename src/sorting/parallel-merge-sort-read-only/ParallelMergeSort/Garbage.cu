float PerformParallelSort_WithoutSharedMemory(int n, int A[], int N, int I, int T, float* t)
{
	float totalTime_p;	
	float totalSortingTime;
	float totalCopyingTime;
	cudaEvent_t* s_evt = new cudaEvent_t[n];
	cudaEvent_t* e_evt = new cudaEvent_t[n];
	cudaEvent_t startEvent_p, endEvent_p;
	cudaEvent_t startCopy, endCopy;
	cudaEvent_t startSort, endSort;
	cudaError_t errorCode;
// Creating events for measuring performance
	cudaEventCreate(&startEvent_p);
	cudaEventCreate(&endEvent_p);
	cudaEventCreate(&startCopy);
	cudaEventCreate(&endCopy);
	cudaEventCreate(&startSort);
	cudaEventCreate(&endSort);
	for (int i = 0; i < n; i++)
	{
		cudaEventCreate(&s_evt[i]);			
		cudaEventCreate(&e_evt[i]);
	}
	errorCode = cudaMemcpy(dev_a, A, sizeof(int)*N, cudaMemcpyHostToDevice);
	ASSERT(errorCode);	

	cudaEventRecord(startEvent_p);

	int* C = new int[N];
	
	// NumberOfElementsPerBlock = Shared memory size / 2
	// Sort<<<NumberOfBlocks, NumberOfThreadsPerBlock, SizeOfSharedMemory>>>
	//		(Source,Destination,NumberOfElementsPerBlock)

	DeviceProperties deviceProps;

	cudaEventRecord(startSort);
	for (int i = 0; i < I; i++)
	{
		int n = 0;
		for (int d = 1; d <= (N/2); d *= 2)
		{
			cudaEventRecord(s_evt[n]);
			int nThreads = min(T, N / (2*d));
			int numberOfBlocks = N / (2*d*nThreads);
			dim3 nBlocks = GetNBlocks(numberOfBlocks, deviceProps.MaxNumberOfBlocks);
			Sort_NonSharedMemory<<<nBlocks,nThreads>>>(dev_a, dev_c, d);
			cudaThreadSynchronize();
			errorCode = cudaGetLastError();
			ASSERT(errorCode);
			
			// Swap contents of dev_c & dev_a
			SwapDeviceArrays();		
			cudaEventRecord(e_evt[n]);
			cudaEventSynchronize(e_evt[n]);
			cudaEventElapsedTime(&t[n], s_evt[n], e_evt[n]);
			n++;
		}
	}
	cudaEventRecord(endSort);
	cudaEventSynchronize(endSort);
	cudaEventElapsedTime(&totalSortingTime, startSort, endSort);
	
	cudaEventRecord(startCopy);
	errorCode = cudaMemcpy(C, dev_a, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaEventRecord(endCopy);
	cudaEventSynchronize(endCopy);
	cudaEventElapsedTime(&totalCopyingTime, startCopy, endCopy);

	cudaEventRecord(endEvent_p);
	cudaEventSynchronize(endEvent_p);
	cudaEventElapsedTime(&totalTime_p, startEvent_p, endEvent_p);

	//cout << totalCopyingTime << " [msec] for copying results to host." << endl;
	delete[] C;
	delete[] s_evt;
	delete[] e_evt;
	return totalTime_p / I;
}
