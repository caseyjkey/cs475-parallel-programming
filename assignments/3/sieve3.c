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
   long BLKSIZE = 100000;

   char *mark;

   long   size;
   long   curr;
   long   i, j,n;
   long   count;

   /* Time */

   double time;

   if ( argc > 1 ) N  = atoi(argv[1]);
   if ( argc > 2 ) BLKSIZE = atoi(argv[2]);

   long p_size = N;

   /* Start Timer */

   initialize_timer ();
   start_timer();

   // Only store odds
   // Half memory if N is even
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

   long sqrt_N = sqrt(p_size);
   for (long k = 3; k <= sqrt_N;) { // Iterate to sqrt(n)
        // We only want to find primes less than sqrt(N)
        // We stride by 2*k because odd + odd = even
        // And odd + even = odd, thereby skipping all even multiples
        // All primes are odd aside from 2
        #pragma omp parallel for
        for (long prime = k*k; prime <= sqrt_N; prime += 2*k) mark[prime/2] = 1;
        
        // Get next odd prime (unmarked value)
        k += 2;
        while (mark[k/2]) k += 2;
   }


   /*number of primes*/
   long primes[sqrt_N]; 
   primes[0] = 2;                   // Don't include 2 because it's even
   count = 1;                       // Count starts from 1 to account for  2
   for(i = 3; i <= sqrt_N; i+=2){   // i starts from 3 as we only count odds
        if(mark[i/2] == 0) {
            primes[count] = i;
            count++;
        }
   }


   /* end of preamble */
   long prime;
   for (int ii = sqrt_N; ii < p_size; ii += BLKSIZE) {
        for (int j = 1; j < count; j++) {  // skip primes[0] because that's evens
           prime = primes[j];
           for (long i = FMIB(ii, prime); i <= minn(ii+BLKSIZE, p_size); i += 2*prime) {
               mark[i/2] = 1;
           }
       }
   } 

   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /*number of primes*/
   // Count and i are magic numbers
   count = 1;
   for(i = 2; i <= N; i+=1){
        if(mark[i] == 0) ++count;
   }

   printf("There are %ld primes less than or equal to %ld\n", count, p_size);
   /* print results */
   printf("First three primes:");
   j = 1;
   printf("%d ", 2);
   for ( i=3 ; i <= p_size; i+=2 ) {
      if (mark[i/2]==0){
            printf("%ld ", i);
            ++j;
      }
   }
   printf("\n");

   printf("Last three primes:");
   j = 0;
   n=(p_size%2?p_size:p_size-1);
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


