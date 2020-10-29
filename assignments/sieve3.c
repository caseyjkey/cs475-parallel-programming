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
    if (index ==  prime)
        return index + prime;
    long remainder = index % prime;
    if (remainder)
        return index - remainder + prime;
    return index;
}

int main(int argc, char **argv) {

   long N  = 100;
   long BLKSIZE = 10;

   char *mark;

   long   size;
   long   curr;
   long   i, j,n;
   long   count;

   /* Time */

   double time;

   if ( argc > 1 ) N  = atoi(argv[1]);
   if ( argc > 2 ) BLKSIZE = atoi(argv[2]);

   /* Start Timer */

   initialize_timer ();
   start_timer();

   // +1 for null character 
   size = (N+1)*sizeof(char);
   mark = (char *)malloc(size);

   for (i=1; i<=N; i=i+1){
     mark[i]=0;
   }

   // We want to skip the value 1 (results start from 2)
   mark[0] = 1;

   long sqrt_N = sqrt(N);
   for (long k = 3; k <= sqrt_N; k += 2) { // Iterate to sqrt(n)
        // We only want to find primes less than sqrt(N)
        // We stride by 2*k because odd + odd = even
        // And odd + even = odd, thereby skipping all even multiples
        // All primes are odd aside from 2
        #pragma omp parallel for
        for (long prime = k*k; prime <= sqrt_N; prime += 2*k) mark[prime] = 1;
        
        // Get next odd prime (unmarked value)
        while (mark[k]) k += 2;
   }


   /*number of primes*/
   long primes[sqrt_N]; // (int *) malloc(count * sizeof(int));
   primes[0] = 2; // Our algorithm doesn't include 2 because it's even
   // Count starts from 1 to include 2
   // i starts from 3 so that we only count odds
   count = 1;
   for(i = 3; i <= sqrt_N; i+=2){
        if(mark[i] == 0) {
            primes[count] = i;
            count++;
        }
   }


   /* end of preamble */

   int prime;
   for (int j = 0; j < count; j++) {
       prime = primes[j];
       for (int ii = sqrt_N; ii < N; ii += BLKSIZE) {
           for (int i = FMIB(ii, prime); i <= minn(ii+BLKSIZE, N); i += prime) {
               mark[i] = 1;
           }
       }
   } 

   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /*number of primes*/
   // Count and i are magic numbers
   count = 1;
   for(i = 3; i <= N; i+=2){
        if(mark[i] == 0) ++count;
   }

   printf("There are %ld primes less than or equal to %ld\n", count, N);
   /* print results */
   printf("First three primes:");
   j = 1;
   printf("%d ", 2);
   for ( i=3 ; i <= N && j < 3; i+=2 ) {
      if (mark[i]==0){
            printf("%ld ", i);
            ++j;
      }
   }
   printf("\n");

   printf("Last three primes:");
   j = 0;
   n=(N%2?N:N-1);
   for (i = n; i > 1 && j < 3; i-=2){
     if (mark[i]==0){
        printf("%ld ", i);
        j++;
     }
   }
   printf("\n");


   printf("elapsed time = %lf (sec)\n", time);

   free(mark);
   return 0;
}


