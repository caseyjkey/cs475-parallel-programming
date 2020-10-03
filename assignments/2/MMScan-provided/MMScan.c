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

#ifdef FAUX  // incorrect parallelization 
#pragma omp parallel for
#endif // incorrect parallelization 
  for(n=start+1; n <= end; n+=1) // 1 to 3
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

void multiplyMatrix(float **X, float **Y, float **T, long size) {
    for(int i=0; i < size; i+=1) {
        for(int j=0; j < size; j+=1) {
            float acc = 0;
            for(int k=0; k<size; k++) {
                acc = acc + Y[i][k] * X[k][j];
            }
            T[i][j] = acc;
        }
    }
    return;
}

void MMScanDNC_helper(float ***X, float ***Y, float ***T, long start, long end, long size, long aux, long p ) {

  if (start == end) {
    for(long i = 0; i < size; i++) {
        for(long j = 0; j < size; j++) {
            T[start][i][j] = X[start][i][j];
            Y[start][i][j] = X[start][i][j];
        }
    }
  }
  else {
    int mid = (start + end) / 2;
    MMScanDNC_helper(X, Y, T, start, mid, size, aux, p);
    MMScanDNC_helper(X, Y, T, mid + 1, end, size, aux, p);
    for(int i = mid + 1; i <= end; i++) {
    	multiplyMatrix(Y[i], Y[mid], T[i], size);
        float **temp = Y[i];
        Y[i] = T[i];
        T[i] = temp;
    }
  }
  return;
}

void MMScanDNC(float ***X, float ***Y, float ***T, long start, long end, long size, long aux, long p ){
 MMScanDNC_helper(X, Y, T, start, end, size, aux, p);
}

void MMScanDNCP1(float ***X, float ***Y, float ***T, long start, long end, long size, long aux, long p ){
  if (start == end) {
    for(long i = 0; i < size; i++) {
        for(long j = 0; j < size; j++) {
            T[start][i][j] = X[start][i][j];
            Y[start][i][j] = X[start][i][j];
        }
    }
  }
  else {
    int mid = (start + end) / 2;
    #pragma omp task if (end-start<p)
    MMScanDNCP1(X, Y, T, start, mid, size, aux, p);

    MMScanDNCP1(X, Y, T, mid + 1, end, size, aux, p);
    
    #pragma omp taskwait
    for(int i = mid + 1; i <= end; i++) {
    	multiplyMatrix(Y[i], Y[mid], T[i], size);
        float **temp = Y[i];
        Y[i] = T[i];
        T[i] = temp;
    }
  }
  return;   
}

void MMScanDNCP2(float ***X, float ***Y, float ***T, long start, long end, long size, long aux, long p ){
  if (start == end) {
    for(long i = 0; i < size; i++) {
        for(long j = 0; j < size; j++) {
            T[start][i][j] = X[start][i][j];
            Y[start][i][j] = X[start][i][j];
        }
    }
  }
  else {
    int mid = (start + end) / 2;
    MMScanDNCP2(X, Y, T, start, mid, size, aux, p);
    MMScanDNCP2(X, Y, T, mid + 1, end, size, aux, p);
    
    #pragma omp parallel for if (end-start>aux)
    for(int i = mid + 1; i <= end; i++) {
    	multiplyMatrix(Y[i], Y[mid], T[i], size);
        float **temp = Y[i];
        Y[i] = T[i];
        T[i] = temp;
    }
  }
  return;   
}

void MMScanDNCP3(float ***X, float ***Y, float ***T, long start, long end, long size, long aux, long p ){
  if (start == end) {
    for(long i = 0; i < size; i++) {
        for(long j = 0; j < size; j++) {
            T[start][i][j] = X[start][i][j];
            Y[start][i][j] = X[start][i][j];
        }
    }
  }
  else {
    int mid = (start + end) / 2;
    #pragma omp task if (end-start<p)
    MMScanDNCP3(X, Y, T, start, mid, size, aux, p);
    MMScanDNCP3(X, Y, T, mid + 1, end, size, aux, p);
    
    #pragma omp taskwait
    #pragma omp parallel for if (end-start>aux)
    for(int i = mid + 1; i <= end; i++) {
    	multiplyMatrix(Y[i], Y[mid], T[i], size);
        float **temp = Y[i];
        Y[i] = T[i];
        T[i] = temp;
    }
  }
  return;   
}
