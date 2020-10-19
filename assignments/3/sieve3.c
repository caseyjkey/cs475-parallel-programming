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

#define minn(x,y) (((x) <= (y)) ? (x) : (y))

int FMIB(long index, long prime) {
    long remainder = index % prime;
    if (remainder)
        return index - remainder + prime;
    return index;
}

int main(int argc, char **argv) {

   long N  = 100;
   long BLKSIZE = 5000;

   char *mark;

   long   size;
   long   curr;
   long   i, j,n;
   long   count;

   /* Time */

   double time;

   if ( argc > 1 ) N  = atoi(argv[1]);
   if ( argc > 2 ) BLKSIZE = atoi(argv[2]);

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

   // +1 for null character 
   size = (N+1)*sizeof(char);
   mark = (char *)malloc(size);

   for (i=1; i<=N; i=i+1){
     mark[i]=0;
   }

   // We want to skip the value 1 (results start from 2)
   mark[0] = 1;

   long sqrt_p_size = sqrt(problem_size);
   for (long k = 3; k <= sqrt_p_size; k += 2) { // Iterate to sqrt(n)
        // We only want to find primes less than sqrt(N)
        // We stride by 2*k because odd + odd = even
        // And odd + even = odd, thereby skipping all even multiples
        // All primes are odd aside from 2
        #pragma omp parallel for
        for (long prime = k*k; prime <= sqrt_p_size; prime += 2*k) mark[prime/2] = 1;
        
        // Get next odd prime (unmarked value)
        while (mark[k/2]) { k += 2; count++; }
   }


   /*number of primes*/
   long primes[sqrt_p_size]; // (int *) malloc(count * sizeof(int));
   primes[0] = 2; // Our algorithm doesn't include 2 because it's even
   
   // Count starts from 1 to include 2
   // i starts from 3 so that we only count odds
   count = 1;
   for(i = 3; i <= sqrt_p_size; i+=2){
        if(mark[i/2] == 0) {
        	printf("\t prime %ld, index %ld \n", i, i/2);
            count++;
            primes[i/2] = i;
        }
   }

   printf("There are %ld primes for sqrt(N).\n", count);
   printf("The primes:\n");
   for(int q = 0; q < count; q++) printf("%ld\n", primes[q]);

   /* end of preamble */

   // +1 for null character 
   size = (problem_size+1)*sizeof(char);
   marked = (char *)realloc(mark, size);

   int prime;
   for (int j = 0; j < count; j++) {
       prime = primes[j];
       for (int ii = sqrt_p_size; ii < problem_size; ii += BLKSIZE) {
           printf("ii: %d\n", ii);
           for (int i = FMIB(ii, prime); i <= minn(ii+BLKSIZE, problem_size); i += prime) {
               printf("FMIB(%d, %d)=%d, i %d, <= %d\n", ii, prime, FMIB(ii, prime), i, minn(ii+BLKSIZE, problem_size)); 
               marked[i] = 1;
           }
       }
   } 


   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /*number of primes*/
   // Count and i are magic numbers
   count = 0;
   for(i = 2; i <= problem_size; i+=1){
        if(marked[i] == 0) {
        	 printf("\t prime %ld  \n",i );
        	++count;
        }

   }
   printf("There are %ld primes less than or equal to %ld\n", count, problem_size);
   /* print results */
   printf("First three primes:");
   j = 1;
   printf("%d ", 2);
   for ( i=3 ; i <= problem_size ; i+=2 ) {
      if (marked[i]==0){
            printf("%ld ", i);
            ++j;
      }
   }
   printf("\n");

   printf("Last three primes:");
   j = 0;
   n=(problem_size%2?problem_size:problem_size-1);
   for (i = n; i > 1 && j < 3; i-=2){
     if (marked[i]==0){
        printf("%ld ", i);
        j++;
     }
   }
   printf("\n");


   printf("elapsed time = %lf (sec)\n", time);

   free(mark);
   return 0;
}


