#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <sys/time.h>

#define min(x,y) ( ((x) < (y))? (x) : (y))


/*
 * Note that the innermost loop can be 0<=j<=i to take 
 * advantage of the symmetry in the input matrix for this
 * kernel. This will only compute half of the output C
 * matrix, saving half the computation. You can try to 
 * tile this version by replacing the upper bound on j
 * with <= N. (This is slightly more complicated.)
 *
 */ 
void kernel_syr2k(int N, int M,
    double C[ N][N],
    double A[ N][M],
    double B[ N][M])
{
  int i, j, k;

  #pragma omp parallel for private(i,j,k)
  for (i = 0; i < N; i++) {
    for (k = 0; k < M; k++) {
      for (j = 0; j < N; j++) {
        C[i][j] += A[j][k] * B[i][k] + B[j][k] * A[i][k];
      }
    }
  }
}


void init_array(int N, int M,
  double C[N][N],
  double A[N][M],
  double B[N][M])
{
  int i, j;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++) {
      A[i][j] = (double) (i*j%N) / N;
      B[i][j] = (double) (i*j%M) / M;
    }
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      C[i][j] = (double) (i*j%N) / M;
}



void print_array(int N,
   double C[N][N])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
 if ((i * N + j) % 20 == 0) fprintf (stderr, "\n");
 fprintf (stderr, "%0.2lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


int main(int argc, char** argv)
{

  int N;
  int M;

  struct timeval t_start;
  struct timeval t_end;
  double etime;

  double* C;
  double* A;
  double* B;

  if (argc < 3) {
    printf("usage ./syr2k N M\n");
    return 0;
  }

  N = atoi(argv[1]);
  M = atoi(argv[2]);

  C = (double*)malloc(N*N * sizeof(double));
  A = (double*)malloc(N*M * sizeof(double));
  B = (double*)malloc(N*M * sizeof(double));

  init_array (N, M, *((double(*)[N][N])C), *((double(*)[N][M])A), *((double(*)[N][M])B));

  gettimeofday (&t_start, NULL);

  kernel_syr2k  (N, M, *((double(*)[N][N])C), *((double(*)[N][M])A), *((double(*)[N][M])B));

  gettimeofday (&t_end, NULL);

  etime = t_end.tv_sec - t_start.tv_sec + 
        (t_end.tv_usec - t_start.tv_usec) * 1.0e-6;

  print_array(N, *((double(*)[N][N])C));

  printf("execution time=%lf\n", etime);

  free(C);
  free(A);
  free(B);

  return 0;
}
