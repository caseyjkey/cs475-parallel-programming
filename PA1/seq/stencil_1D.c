/*
 * 5 point stencil 
 *
 *  Created on: Sep 12, 2010
 *      Author: Wim Bohm
 */

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define   INIT_VALUE       5000

void printResult(double *data, int size);

int main(int argc, char **argv) {

   int     N;
   int     t;
   int     MAX_ITERATION = 2000;
   double  *prev, *cur;
   double  error = INIT_VALUE;

   // Timer
   double  time;

   // temporary variables
   int     i,j;
   double  *temp;

   // Check commandline args.
   if ( argc > 1 ) {
      N = atoi(argv[1]);
   } else {
      printf("Usage : %s [N]\n", argv[0]);
      exit(1);
   }
   if ( argc > 2 ) {
      MAX_ITERATION = atoi(argv[2]);
   }

   // Memory allocation for data array.
   prev  = (double *) malloc( sizeof(double) * N);
   cur   = (double *) malloc( sizeof(double) * N);
   if ( prev == NULL || cur == NULL ) {
      printf("[ERROR] : Fail to allocate memory.\n");
      exit(1);
   }

   // Initialization
   for ( i=1 ; i < N-1 ; i++ ) {
         prev[i] = 0.0;
      }

      prev[0]  = prev[1]  = INIT_VALUE;
      prev[N-1]  = prev[N-2]  = INIT_VALUE;
      cur[0]  = cur[1]  = INIT_VALUE;
      cur[N-1]  = cur[N-2]  = INIT_VALUE;

   initialize_timer();
   start_timer();

   // Computation
   t = 0;


   while ( t < MAX_ITERATION) {

      // Computation
      for ( i=2 ; i < N-2 ; i++ ) {
            cur[i] = (prev[i-2]+prev[i-1]+prev[i]+prev[i+1]+prev[i+2])/5;
       }

      {
      temp = prev;
      prev = cur;
      cur  = temp;
      t++;
      }
   }

   stop_timer();
   time = elapsed_time();

   printResult(prev, N);

   printf("Data size : %d  , #iterations : %d , time : %lf sec\n", N, t, time);
}

void printResult(double *data, int size) {
   int i;

/* print a portion of the vector */
     printf("data[%d]: %lf \n",0, data[0]);
     printf("data[%d]: %lf \n",1, data[1]);
     printf("data[%d]: %lf \n",size/10, data[size/10]);
     printf("data[%d]: %lf \n",size/5, data[size/5]);
     printf("data[%d]: %lf \n",size/2, data[size/2]);
  
   return;
}

