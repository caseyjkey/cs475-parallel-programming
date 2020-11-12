/*/////////////////////////////////////////////////////////////////////////////
//
// File name : MMScan.c
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

void MMScan(float ***X, float ***Y, long start, long end, long size){
  long n, i, j, k;
  for(i=0; i <= size-1; i+=1)
    {
      for(j=0; j <= size-1; j+=1)
	{
	  Y[start][i][j] = X[start][i][j];
	}
    }

  for(n=start+1; n <= end; n+=1)
    {
      for(i=0; i < size; i+=1)
	{
	  for(j=0; j < size; j+=1)
	    {
	      float acc = 0;
	      for(k=0; k<size; k++){
		acc = acc + Y[n-1][i][k] * X[n][k][j];
	      }
	      Y[n][i][j] = acc;
	    }
	}
    }
}
