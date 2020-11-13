#include "MMScanKernel.h"

// Find N/G breakpoints for our scan
// G groups/blocks
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){  long matrixSize = sizeof(float)*B*B;
  extern __shared__ float array[];
	
	// B0, B1 will hold intermediate products
	// B2 will hold results
  float* B0 = array;
  float* B1 = &array[B*B];
  float* B2 = &array[2*B*B];

  // Do calculation
	long X, Y, X_offset, Y_offset, P;
  long matricesPerBlock = N/G;
	long numMatrices = blockIdx.x * matricesPerBlock;
	long elementsPerMatrix = B * B;
	long result;

  for (long Q = 0; Q < matricesPerBlock; Q++) {
    long currentMatrix = Q * elementsPerMatrix;
    long chain_offset = numMatrices * elementsPerMatrix + currentMatrix;
		for (Y = 0; Y < B/S; Y++) {
      Y_offset = Y * B * S + threadIdx.y * B;
      for (X = 0; X < B/S; X++) {
      	X_offset = X * S + threadIdx.x;
        B1[Y_offset + X_offset] = X_GPU[chain_offset + Y_offset + X_offset];
      }
    }
		
		for (Y = 0; Y < B/S; Y++) {
			Y_offset = Y * B * S + threadIdx.y * B;		
			for (X = 0; X < B/S; X++) {
				X_offset = X * S + threadIdx.x * B;
				result=0;
				for (P=0; P < B; P++) {
					result += B0[Y_offset + P] * B1[X_offset + P * B];
				}			
				B2[X_offset + Y_offset] = result;
			}
		}
  }
  // Save result to device memory
  __syncthreads();

  //X_GPU = B2;
  memcpy(R1_GPU, B2, sizeof(float)*B*B);

  return;
}
