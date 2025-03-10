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
   int start = (n/p) * id - k;
		
	 int length = n/p + 2*k;

	 for ( i=start ; i < start+length; i++ )
			prev[i-start] = i;


   cur[0] = 0;
	 cur[length-1] = start + length - 1;
	 
	 if(v){
		 printf("\n---------------- INIT id: %d -------------\n", id);
		 printf("%d: prev", id);
     for(i=k;i<length-k;i++) printf(" %f ", id, prev[i]);
		 printf("\n%d: cur", id);
		 for(i=k;i<length-k;i++) printf(" %f ", id, cur[i]);
		 printf("\n\n");
		 printf("--------------- END INIT ------------------\n");   
	 }

   // Wait for all processes are ready before starting timer
   MPI_Barrier(MPI_COMM_WORLD);   

   initialize_timer();
   start_timer();

   // Computation
   t = 0;
   int stop;
   
   if ( id != p-1 ) 
      stop = n/p + 2*k; 
   else 
      stop = n/p + k-1;

  if ( id == 0 )
		i = 1+k; 
	else
		i = 1;

	while ( t < m) {
			if (v)
				printf("------------- t = %d --------------\n", t);
      for ( ; i < stop ; i++ ) {

         cur[i] = (prev[i-1]+prev[i]+prev[i+1])/3;
				 if (v)
						printf("proc %d cur[%d]: %f = (%f + %f + %f)/3\n", id, i, cur[i], prev[i-1], prev[i], prev[i+1]); 
			}
      temp = prev;
      prev = cur;
      cur  = temp;
      t++;
      // Refresh buffers
      if (t % k == 0) {
			   // send right side of array to right neighbor
				 // load right buffer
         if ( id != p-1 ) {
				    MPI_Send(prev+(n/p), k, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);	
            MPI_Recv(cur+k+(n/p), k, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
         }
				 // send left side of array to left neighbor
				 // load left buffer
         if ( id != 0 ) {
            MPI_Send(prev+k, k, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);	
            MPI_Recv(cur, k, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
         }
         MPI_Barrier(MPI_COMM_WORLD);
      }  
   }

	
	 if(v){
		 MPI_Barrier(MPI_COMM_WORLD);
		 printf("%d: prev", id);
     for(i=k;i<length-k;i++) printf(" %f ", id, prev[i]);
		 printf("\n%d: cur", id);
		 for(i=k;i<length-k;i++) printf(" %f ", id, cur[i]);
		 printf("\n\n");
	 }
	
   if (t % 2 == 0) {
		cur[length-k-1] = prev[length-k-1];
	 }

   MPI_Gather(cur+k, n/p, MPI_DOUBLE, end, n/p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   stop_timer();
   time = elapsed_time();
	 if (id == 0) {
		 if(v){
			 for(i=0;i<n;i++) printf("%f ", end[i]);
			 printf("\n");
		 }
		 else
			 printf("first, mid, last: %f %f %f\n",end[0], end[n/2-1], end[n-1]);

    printf("Data size : %d  , #iterations : %d , time : %lf sec\n", n, t, time);
	}
     
	MPI_Finalize();
}



