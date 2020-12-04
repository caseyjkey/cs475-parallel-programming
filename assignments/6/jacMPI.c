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
   double  *start, *prev, *cur, *end;

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
   int arrSize = subSize + (2 * b); 
   end   = (double *) malloc( sizeof(double) * n ); 
   prev  = (double *) malloc( sizeof(double) * arrSize );
   cur   = (double *) malloc( sizeof(double) * arrSize );
   if ( prev == NULL || cur == NULL ) {
      printf("[ERROR] : Fail to allocate memory.\n");
      exit(1);
   }

   // Initialization

   if ( id == 0 ) {
      for ( i=0 ; i < arrSize-b ; i++ ) {
         prev[i] = i;
      }
      cur[arrSize-b-1] = arrSize-b-1;
   }
   else if ( id == p-1 ) {
      for ( i=0 ; i < arrSize-b ; i++ ) {
         prev[i] = (n/p*id) - b + i;
      }
      cur[arrSize-b-1] = (n/p*id) - b + arrSize-b-1;
   }
   else {
      for ( i=0 ; i < arrSize ; i++ ) {
         prev[i] = (n/p*id) - b + i;
      }
      cur[arrSize-1] = (n/p*id) - b + arrSize-1;
   }

   cur[0]  = 0;

      

   // Wait for all processes are ready before starting timer
   MPI_Barrier(MPI_COMM_WORLD);   

   initialize_timer();
   start_timer();

   // Computation
   t = 0;
   
   if ( id == 0 ) {
      while ( t < m) {
         for ( i=1 ; i < subSize+b-1 ; i++ ) {
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
            for ( i=1 ; i < subSize+b-1 ; i++ ) {
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

   MPI_Gather();

   stop_timer();
   time = elapsed_time();

   if(v){
     for(i=0;i<n;i++) printf("%f ",prev[i]);
     printf("\n");
   }
   else
     printf("first, mid, last: %f %f %f\n",prev[0], prev[n/2-1], prev[n-1]);
     
   printf("Data size : %d  , #iterations : %d , time : %lf sec\n", n, t, time);
}



