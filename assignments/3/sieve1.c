/*/////////////////////////////////////////////////////////////////////////////
//
// File name : sieve.c
// Author    : Nissa Osheim
// Date      : 2010/19/10
// Desc      : Finds the primes up to N
//
// updated Wim Bohm
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

int main(int argc, char **argv) {

   long N  = 100;

   char *mark;

   long   size;
   long   curr;
   long   i, j,n;
   long   count;

   /* Time */

   double time;

   if ( argc > 1 ) N  = atoi(argv[1]);

   // N will get cut in half to save memory
   long problem_size = N;

   /* Start Timer */

   initialize_timer ();
   start_timer();

   // Only store odds
   // Half memory if N % 2 == 0
   // Half memory +1 if N is odd
   if (N % 2 == 0)
       N = (N/2);
   else
       N = (N/2 + 1);

   // +1 for null character (but it's not a string?) 
   size = (N+1)*sizeof(char);
   mark = (char *)malloc(size);

   for (i=1; i<=N; i=i+1){
     mark[i]=0;
   }

   // We want to skip the value 1 (results start from 2)
   mark[0] = 1;

   int k, index;
   k = 3; index = 3;

   while (k*k <= problem_size) { // Iterate to sqrt(n)
        for (i = k*k; i <= problem_size; i += 2*k) mark[i/2] = 1;
        
        // Get next odd prime (unmarked value)
        index += 2;
        while (mark[index/2]) index += 2; 
        k = index;
   }

   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /*number of primes*/
   // Count and i are magic numbers
   count = 1;
   for(i = 2; i <= N; i+=1){
        if(mark[i] == 0) {
        	//  printf("\t prime %ld  \n",i );
        	++count;
        }

   }
   printf("There are %ld primes less than or equal to %ld\n", count, problem_size);
   /* print results */
   printf("First three primes:");
   j = 1;
   printf("%d ", 2);
   for ( i=3 ; i <= problem_size && j < 3; i+=2 ) {
      if (mark[i/2]==0){
            printf("%ld ", i);
            ++j;
      }
   }
   printf("\n");

   printf("Last three primes:");
   j = 0;
   n=(problem_size%2?problem_size:problem_size-1);
   for (i = n; i > 1 && j < 3; i-=2){
     if (mark[i/2]==0){
        printf("%ld ", i);
        j++;
     }
   }
   printf("\n");


   printf("elapsed time = %lf (sec)\n", time);

   free(mark);
   return 0;
}


