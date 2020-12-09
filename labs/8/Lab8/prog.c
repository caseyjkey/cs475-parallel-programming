/*/////////////////////////////////////////////////////////////////////////////
//
// File name  : prog.c
// Author     : Swetha Varadarajan
// Description:  wavefront paralellization for openmp.
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "timer.h"

#define max(x,y)   	((x)>(y) ? (x) : (y))
#define min(x,y)   	((x)>(y) ? (y) : (x))
#define foo(x,y)   	((x)+(y))/2
#define A(i,j)	   	A[((i)*(N))+(j)]

int main(int argc, char **argv) {

	long N = 100; 
	int verbose = 0;
	long i,j,p,t;
	long* A;
	double time;
	
	if (argc > 1) N  = atoi(argv[1]);
	if (argc > 2) verbose = 1;
	printf("N=%ld\n", N);
	
	A = (long *)malloc(N*N*sizeof(long));
	
	if (NULL == A) {
		fprintf(stderr, "malloc failed\n");
		return(-1);
	}
	
	/* Initialization */
	A(0,0) = 100;
	for (i=1; i<N; i++) {
		A(0,i)=foo(A(0,i-1),i);
		A(i,0)=foo(i,A(i-1,0));
	}

	/* Start Timer */
	initialize_timer();
	start_timer();
	
	#ifdef SEQ
	for (i=1; i<N; i++)
		for (j=1; j<N; j++) {
			A(i,j) = foo(A(i,j-1), A(i-1,j));
		}
	#endif
	
	#ifdef PAR
	/*Applying the transformation: (i,j->p,t) = (i,j->i,i+j-1) from lecture slides

	        j                                       t
	    0 1 2 3 4                             0 1 2 3 4 5 6 7 8 
	  0 x-x-x-x-x-->                        0 ┌------------------> 
	  1 x * * * *                           1 | * * * *

	  2 x * * * *        (i,j -> p,t)       2 |   * * * * 

	i 3 x * * * *      --------------->   p 3 |     * * * *
                                                      ^ ^ ^
	  4 x * * * *                           4 |       *<*<*<*
	  5 |                                   5 | 


	 The dependencies in the original (left) space point west and north.
	 The dependencies in the transformed (right) space point west and northwest.
	
	 Before the transformation, no points for a given i or j can be done in parallel.
	 After the transformation, ALL points for a given t can now be done in parallel.
	 Each t represents a wavefront, and now all points within a wavefront (i.e. column
	   in the right image), can be done in parallel.

	 With an outer loop on t, then we can parallelize the inner p-loop.

	 Notice that the whole reason you need to do this is because it is not legal to
	   parallelize either the i- or j-loop in the above SEQ version as written.

	 Loop body should express i and j in t and p, that is derive expressions for i 
	   and j in t and p.
	*/
	for (t=1; t<2*N-2; t++)
        #pragma omp parallel for
		for (p=max(1,-N+t+2); p<=min(N-1,t); p++) {
			// these are currently incorrect, you need to change them
			i = p;
			j = 1+t-p;
			A(i,j) = foo(A(i,j-1), A(i-1,j));
		}
	#endif
	
	/* Stop Timer */
	stop_timer();
	time = elapsed_time();
	
	if(verbose) {
		for (i=0; i<N; i++) {
			for (j=0; j<N; j++)
				printf("A(%ld,%ld)=%ld\t", i, j, A(i,j));
			printf("\n");
		}
	}
	 
	printf("elapsed time = %lf\n", time);
	free(A);
	return 0;
}
