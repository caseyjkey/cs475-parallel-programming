#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <sys/time.h>

#define min(x,y) ( ((x) < (y))? (x) : (y))


void kernel_mvt(int N,
  double x1[N],
  double x2[N],
  double y_1[N],
  double y_2[N],
  double A[N][N])
{
  int i, j;

  for (i = 0; i <N; i++)
    for (j = 0; j <N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i <N; i++)
    for (j = 0; j <N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];

}

void init_array(int N,
  double x1[ N],
  double x2[ N],
  double y_1[ N],
  double y_2[ N],
  double A[ N][N])
{
  int i, j;

  for (i = 0; i < N; i++)
    {
      x1[i] = (double) (i % N) / N;
      x2[i] = (double) ((i + 1) % N) / N;
      y_1[i] = (double) ((i + 3) % N) / N;
      y_2[i] = (double) ((i + 4) % N) / N;
      for (j = 0; j < N; j++)
 A[i][j] = (double) (i*j % N) / N;
    }
}

void print_array(int N,
   double x1[N],
   double x2[N])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x1");
  for (i = 0; i < N; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.2lf ", x1[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x1");

  fprintf(stderr, "begin dump: %s", "x2");
  for (i = 0; i < N; i++) {
    if (i % 20 == 0) fprintf (stderr, "\n");
    fprintf (stderr, "%0.2lf ", x2[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x2");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}



int main(int argc, char** argv)
{

  int N;

  struct timeval t_start;
  struct timeval t_end;
  double etime;

  double* A;
  double* x1;
  double* x2;
  double* y_1;
  double* y_2;

  if (argc < 2) {
    printf("usage ./mvt N\n");
    return 0;
  }

  N = atoi(argv[1]);

  A   = (double*)malloc(N*N * sizeof(double));
  x1  = (double*)malloc(N   * sizeof(double));
  x2  = (double*)malloc(N   * sizeof(double));
  y_1 = (double*)malloc(N   * sizeof(double));
  y_2 = (double*)malloc(N   * sizeof(double));

  init_array (N,
       *((double(*)[N])x1),
       *((double(*)[N])x2),
       *((double(*)[N])y_1),
       *((double(*)[N])y_2),
       *((double(*)[N][N])A));

  gettimeofday (&t_start, NULL);

  kernel_mvt (N,
       *((double(*)[N])x1),
       *((double(*)[N])x2),
       *((double(*)[N])y_1),
       *((double(*)[N])y_2),
       *((double(*)[N][N])A));

  gettimeofday (&t_end, NULL);

  etime = t_end.tv_sec - t_start.tv_sec + 
        (t_end.tv_usec - t_start.tv_usec) * 1.0e-6;

  print_array(N, *((double(*)[N])x1), *((double(*)[N])x2));

  printf("execution time=%lf\n", etime);

  free((void*)A);;
  free((void*)x1);;
  free((void*)x2);;
  free((void*)y_1);;
  free((void*)y_2);;

  return 0;
}
