#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <sys/time.h>

#define min(x,y) ( ((x) < (y))? (x) : (y))

void kernel_trmm(int M, int N,
   double A[M][M],
   double B[M][N])
{
  int i, j, k;

  for (i = 0; i < M; i++)
     for (j = 0; j < N; j++) {
        for (k = i+1; k < M; k++)
           B[i][j] += A[k][i] * B[k][j];
     }

}


void init_array(int M, int N,
  double A[ M][M],
  double B[ M][N])
{
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (double)((i+j) % M)/M;
    }
    A[i][i] = 1.0;
    for (j = 0; j < N; j++) {
      B[i][j] = (double)((N+(i-j)) % N)/N;
    }
 }

}

void print_array(int M, int N,
   double B[M][N])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "B");
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) {
 if ((i * M + j) % 20 == 0) fprintf (stderr, "\n");
 fprintf (stderr, "%0.2lf ", B[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "B");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

int main(int argc, char** argv)
{

  int N;
  int M;

  struct timeval t_start;
  struct timeval t_end;
  double etime;

  double* A;
  double* B;

  if (argc < 3) {
    printf("usage ./trmm N M\n");
    return 0;
  }

  N = atoi(argv[1]);
  M = atoi(argv[2]);

  A = (double*)malloc(M*M * sizeof(double));
  B = (double*)malloc(M*N * sizeof(double));

  init_array (M, N, *(double(*)[M][M])A, *(double(*)[M][N])B);

  gettimeofday (&t_start, NULL);

  kernel_trmm (M, N, *(double(*)[M][M])A, *(double(*)[M][N])B);

  gettimeofday (&t_end, NULL);

  etime = t_end.tv_sec - t_start.tv_sec + 
        (t_end.tv_usec - t_start.tv_usec) * 1.0e-6;

  print_array(M, N, *(double(*)[M][N])B);

  printf("execution time=%lf\n", etime);

  free((void*)A);
  free((void*)B);

  return 0;
}
