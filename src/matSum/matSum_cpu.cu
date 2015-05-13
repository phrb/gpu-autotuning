#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


void matSum(float* S, float* A, float* B, unsigned int N) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      int tid = i*N + j;
      S[tid] = A[tid] + B[tid];
    }
  }
}


// Fills a vector with random float entries.
void randomInit(float* data, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int tid = i*N+j;
      data[tid] = (float)drand48();
    }
  }
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    fprintf(stderr, "Syntax: %s <matrix N>\n", argv[0]);
    return EXIT_FAILURE;
  }
  int N = atoi(argv[1]);

  // allocate host memory for matrices A and B
  printf("Allocate memory for matrices A and B...\n");
  float* A = (float*) malloc(N * N * sizeof(float));
  float* B = (float*) malloc(N * N * sizeof(float));
  float* S = (float*) malloc(N * N * sizeof(float));

  // set seed for drand48()
  srand48(42);

  // initialize matrices
  printf("Initialize matrices...\n");
  randomInit(A, N);
  randomInit(B, N);

  printf("Sum matrices...\n");
  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  matSum( S, A, B, N );
  gettimeofday(&end, NULL);

  double cpuTime = 1000000*(double)(end.tv_sec - begin.tv_sec);
  cpuTime +=  (double)(end.tv_usec - begin.tv_usec);

  // print times
  printf("\nExecution Time (microseconds): %9.2f\n", cpuTime);

  // clean up memory
  free(A);
  free(B);
  free(S);

  return 0;
}

