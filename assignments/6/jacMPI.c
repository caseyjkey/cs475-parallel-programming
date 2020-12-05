/*
 * jacobi.c
 * WimBo
 */

#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include "timer.h"

int main(int argc, char **argv) {

   int     id; // process id
   int     p; // number of processes
   int     k; // buffer size   
   int     n; // problem size
   int     t; // loop var for number of iterations
   int     m = 2000; // number of iterations
   double  *prev, *cur, *end;

   MPI_Status status; // return status for receive

   MPI_Init( &argc, &argv );
   MPI_Comm_rank( MPI_COMM_WORLD, &id );
   MPI_Comm_size( MPI_COMM_WORLD, &p );


   // Timer
   double  time;

   // temporary variables
   int i,j;
   int v = 0; //verbose
   double  *temp;

   // Check commandline args.
   if ( argc > 1 ) {
      n = atoi(argv[1]);
   } else {
      printf("Usage : %s [N]\n", argv[0]);
      exit(1);
   }
   if ( argc > 2 ) {
      m = atoi(argv[2]);
   }
   if ( argc > 3 ) {
      k = atoi(argv[3]);
   }
   if ( argc > 4 ) {
      v = 1;
   }

   // Memory allocation for data array.
   int subSize  = n / p;
   int arrSize = subSize + (2 * k); 
   end   = (double *) malloc( sizeof(double) * n ); 
   prev  = (double *) malloc( sizeof(double) * arrSize );
   cur   = (double *) malloc( sizeof(double) * arrSize );
   if ( prev == NULL || cur == NULL ) {
      printf("[ERROR] : Fail to allocate memory.\n");
      exit(1);
   }

   // Initialization
   int start;
	 if ( id == 0 )
			start = 0;
	 else
		  start = (n/p) * id - k;
		
	 int length;
	 if ( id == 0 || id == p-1 )
			length = n/p + k;
	 else
			length = n/p + 2*k;

	 for ( i=start ; i < length; i++ )
			prev[i-start] = i;

	 if ( id == p-1 )
      cur[arrSize-k-1] = (n/p*id) - k + arrSize-k-1;

   cur[0] = 0;

	 if(v){
		 printf("\n---------------- INIT id: %d -------------\n", id);
		 printf("----- prev ----\n");
     for(i=0;i<length;i++) printf("%f ", prev[i]);
     printf("\n");
		 printf("----- cur -----\n");
		 for(i=0;i<length;i++) printf("%f ", cur[i]);
     printf("\n");
		 printf("--------------- END INIT ------------------\n");
   }

      

   // Wait for all processes are ready before starting timer
   MPI_Barrier(MPI_COMM_WORLD);   

   initialize_timer();
   start_timer();

   // Computation
   t = 0;
   
   if ( id == 0 ) {
      while ( t < m) {
         for ( i=1 ; i < subSize+k-1 ; i++ ) {
               cur[i] = (prev[i-1]+prev[i]+prev[i+1])/3;
          }
         temp = prev;
         prev = cur;
         cur  = temp;
         t++;
      }
   }
   else if ( id == p-1 ) {
      while ( t < m) {
            for ( i=1 ; i < subSize+k-1 ; i++ ) {
                  cur[i] = (prev[i-1]+prev[i]+prev[i+1])/3;
             }
            temp = prev;
            prev = cur;
            cur  = temp;
            t++;
      }
   }
   else {
      while ( t < m) {
            for ( i=1 ; i < arrSize-1 ; i++ ) {
                  cur[i] = (prev[i-1]+prev[i]+prev[i+1])/3;
             }
            temp = prev;
            prev = cur;
            cur  = temp;
            t++;
       
      }
   }

   MPI_Gather(prev+k, n/p, MPI_DOUBLE, end, n/p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   stop_timer();
   time = elapsed_time();

   if(v){
     for(i=0;i<n;i++) printf("%f ", end[i]);
     printf("\n");
   }
   else
     printf("first, mid, last: %f %f %f\n",prev[0], prev[n/2-1], prev[n-1]);
     
   printf("Data size : %d  , #iterations : %d , time : %lf sec\n", n, t, time);
}



