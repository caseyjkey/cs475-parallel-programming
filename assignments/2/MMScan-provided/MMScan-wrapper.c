/*/////////////////////////////////////////////////////////////////////////////
//
// File name : MMScan_wrapper.c
// Author    : Sanjay Rajopdhye
// Date      : 2019/Sept/16
// Desc      : Finds the prefix product of an array of BxB matrices
//
/////////////////////////////////////////////////////////////////////////////*/

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <omp.h>

#define max(x, y)   ((x)>(y) ? (x) : (y))
#define min(x, y)   ((x)>(y) ? (y) : (x))

// Common Macros
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }
#define EPSILON 1.0E-6

void MMScan(float***, float***, long, long, long);
void MMScanDNC(float***, float***, float***, long, long, long, long, long);
void MMScanDNCP1(float***, float***, float***, long, long, long, long, long);
void MMScanDNCP2(float***, float***, float***, long, long, long, long, long);
void MMScanDNCP3(float***, float***, float***, long, long, long, long, long);

//main
int main(int argc, char** argv) {
  //Check number of args
  if (argc <= 2) {
    printf("Number of argument is smaller than expected.\n");
    printf("Expecting N,B\n");
    exit(0);
  }

  char *end = 0;
  char *val = 0;
  //Read Parameters
  //Initialization of N
  val = argv[1];
  long N = atoi(val);

  //Initialization of B
  val = argv[2];
  long B = atoi(val);

  long tuning = 0;
  //Additional args?
  if(argc > 3)
    {
      val = argv[3];
      tuning = atoi(val);
    }

  ///Parameter checking
  if (!((N >= 1 && B >= 1))) {
    printf("The value of parameters are not valid.\n");
    exit(-1);
  }
  
  //Memory Allocation
  long n, i, j, k;
  float* _lin_X = (float*)malloc(sizeof(float)*((N) * (B) * (B)));
  mallocCheck(_lin_X, ((N) * (B) * (B)), float);
  float*** X = (float***)malloc(sizeof(float**)*(N));
  mallocCheck(X, (N), float**);
  for (n=0;n < N; n++) {
    X[n] = (float**)malloc(sizeof(float*)*(B));
    mallocCheck(X[n], (B), float*);
    for (i=0;i < B; i++) {
      X[n][i] = &_lin_X[(n*((B) * (B))) + (i*(B))];
    }
  }

  float* _lin_Y = (float*)malloc(sizeof(float)*((N) * (B) * (B)));
  mallocCheck(_lin_Y, ((N) * (B) * (B)), float);
  float*** Y = (float***)malloc(sizeof(float**)*(N));
  mallocCheck(Y, (N), float**);
  for (n=0;n < N; n++) {
    Y[n] = (float**)malloc(sizeof(float*)*(B));
    mallocCheck(Y[n], (B), float*);
    for (i=0;i < B; i++) {
      Y[n][i] = &_lin_Y[(n*((B) * (B))) + (i*(B))];
    }
  }

  float* _lin_Temp = (float*)malloc(sizeof(float)*((N) * (B) * (B)));
  mallocCheck(_lin_Temp, ((N) * (B) * (B)), float);
  float*** Temp = (float***)malloc(sizeof(float**)*(N));
  mallocCheck(Temp, (N), float**);
  for (n=0;n < N; n++) {
    Temp[n] = (float**)malloc(sizeof(float*)*(B));
    mallocCheck(Temp[n], (B), float*);
    for (i=0;i < B; i++) {
      Temp[n][i] = &_lin_Temp[(n*((B) * (B))) + (i*(B))];
    }
  }

  //Initialization of rand
  srand((unsigned)time(NULL));

  //Input Initialization
  
#if defined (RANDOM)
  float x, y, tmp;
  x = (float) rand();
  for(n=0; n <= N-1; n+=1)
    {
      y = (float) rand();
      for(i=0; i <= B-1; i+=1)
	for(j=0; j <= B-1; j+=1)
	  X[n][i][j] = y/(B*x);
      x = y;
    }
#else  // not random
  for(i=0; i <= B-1; i+=1)
    for(j=0; j <= B-1; j+=1)
      X[0][i][j] = (float) 1.0;   // all 1s
  for(n=1; n <= N-1; n+=1)
  {
      for(i=0; i <= B-1; i+=1)
	{
	  for(j=0; j <= B-1; j+=1)
	    {
#if defined (INTERACTIVE)
	      {
		printf("X[%ld][%ld][%ld]= ", n, i, j);
		scanf("%f", &X[n][i][j]);
	      }
#else // neither random not interactive, i.e., default
	      X[n][i][j] = (float) (n+1)/((float) (B*n));
#endif
	    }
	}
  }

#endif
  
  //Timing
  struct timeval time;
  double elapsed_time1, elapsed_time2;

  //Call the main computation

  //**************************************************************************//
  /*                     START OF THE SCAN COMPUTATION                        */
  //**************************************************************************//
  /* int p = omp_get_num_procs(); */
  /* printf("There are %ld threads\n", p); */

  gettimeofday(&time, NULL);
  elapsed_time1 = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

#if defined DNC  
  MMScanDNC(X, Y, Temp, 0, N-1, B, 10000, tuning); // last param is from command
					      // line and the one before that
					      // is arbitrary
#elif defined DNCP1
  #pragma omp parallel
  {
        #pragma omp single
        MMScanDNCP1(X, Y, Temp, 0, N-1, B, 10000, tuning); // last param is from command
  }
					      // line and the one before that
	
#elif defined DNCP2
  #pragma omp parallel
  {
        #pragma omp single
        MMScanDNCP2(X, Y, Temp, 0, N-1, B, tuning, 1); // last param is from command
  }

#elif defined DNCP3
  #pragma omp parallel
  {
        #pragma omp single
        MMScanDNCP3(X, Y, Temp, 0, N-1, B, 10000, tuning); // last param is from command
  }



#else
  MMScan(X, Y, 0, N-1, B);
#endif
  gettimeofday(&time, NULL);
  elapsed_time1 = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time1;

  float ***tmp_ptr = Temp;
  Temp = Y;  Y = tmp_ptr; // swap Temp and Y so that the next call computes Y
			  // with the standard sequential algorithm

  gettimeofday(&time, NULL);
  elapsed_time2 = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

  // the provided seqential algorithm
  
  MMScan(X, Y, 0, N-1, B);

  gettimeofday(&time, NULL);
  elapsed_time2 = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time2;

  //**************************************************************************//
  /*                       END OF THE SCAN COMPUTATION                        */
  /*                                                                          */
  /*                    PRINT OUTPUTS (DEPENDING ON FLAGS)                    */
  //**************************************************************************//
    
#ifdef INTERACTIVE
  //Print Outputs Interactively
	
  for(n=0; n <= N-1; n+=1)
    {
      printf("Y[%ld][i][j]= \n", n);
      for(i=0; i <= B-1; i+=1)
	{
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", Y[n][i][j]);
	    }
	  printf("\n");	  
	}
      printf("\n");
    }
#endif

#ifdef VERBOSE
  //Print Inputs and Outputs (leading and trailing, no more than 5 each)

  //  First print the first five (X, Y and Temp)
  for(n=0; n <= min(N-1, 5); n+=1)
    {
      printf("\tX[%ld][i][j], \tY[%ld][i][j], \tTemp[%ld][i][j] \n", n, n, n);
      for(i=0; i <= B-1; i+=1)
	{
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", X[n][i][j]);
	    }
	  printf("\t");	  
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", Y[n][i][j]);
	    }
	  printf("\t");	  
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", Temp[n][i][j]);
	    }
	  printf("\n");
	}
      printf("\n");
    }

  //  Then print the last five (X, Y and Temp)
  for(n=max(5, N-5); n <= N-1; n+=1)
    {
      printf("\tX[%ld][i][j], \tY[%ld][i][j], \tTemp[%ld][i][j] \n", n, n, n);
      for(i=0; i <= B-1; i+=1)
	{
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", X[n][i][j]);
	    }
	  printf("\t");	  
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", Y[n][i][j]);
	    }
	  printf("\t");	  
	  for(j=0; j <= B-1; j+=1)
	    {
	      printf("%10g ", Temp[n][i][j]);
	    }
	  printf("\n");	  
	}
      printf("\n");
    }
#endif

#if defined CHECKING
  // Compare the values in Y and Temp and count how many are different
  long error_count = 0;

  for(n=0; n <= N-1; n+=1)
    {
      for(i=0; i <= B-1; i+=1)
	{
	  for(j=0; j <= B-1; j+=1)
	    {
	      if (fabs(Temp[n][i][j]-Y[n][i][j]) > EPSILON)
		{error_count += 1;
		  printf ("Temp[%ldl][%ldl][%ldl] = %f, \tY[%ldl][%ldl][%ldl] = %f\n",
			  n, i, j, Temp[n][i][j], n, i, j, Y[n][i][j]);
		}
	    }
	}
    }
  printf("The total number of errors is %ld\n", error_count);
#endif 

  // timing information
  
  printf("Execution time for DNC:\t%lf sec.\n", elapsed_time1);	
  printf("Execution time for SEQ:\t%lf sec.\n", elapsed_time2);
    
  //Memory Free
  free(_lin_X);
  for (n=0;n < N; n++) {
    free(X[n]);
  }
  free(X);
  free(_lin_Y);
  for (n=0;n < N; n++) {
    free(Y[n]);
  }
  free(Y);
	
  return EXIT_SUCCESS;
}

//Common Macro undefs
#undef EPSILON
