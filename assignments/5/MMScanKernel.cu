#include "MMScanKernel.h"

__device__ void swapArray(float* source, float* dest, int size) {
	for(int i = 0; i < size*size; i++) {
		source[i] = source[i] + dest[i];
		dest[i] = source[i] - dest[i];
		source[i] = source[i] - dest[i];	
	}
}

// Find N/G breakpoints for our scan
// G groups/blocks
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){  long matrixSize = sizeof(float)*B*B;
	long subMatrixDim = (long) S;
  extern __shared__ float array[];
	
	// B0, B1 will hold intermediate products
	// B2 will hold results
  float* B0 = array;
  float* B1 = &array[B*B];
  float* B2 = &array[2*B*B];

  // Do calculation
	long X, Y, X_offset, Y_offset, P;
  long matricesPerBlock = N/gridDim.x;
	long numMatrices = blockIdx.x * matricesPerBlock;
	long elementsPerMatrix = B * B;
	long result;

	// Initialize B0
	// If slow, use different threads to do this
	memset(B0, 0, sizeof(float) * B * B);
	for (Y = 0; Y < B; Y++)
		B0[Y * B + Y] = 1;

  for (long Q = 0; Q < matricesPerBlock; Q++) {
    long currentMatrix = Q * elementsPerMatrix;
    long chain_offset = numMatrices * elementsPerMatrix + currentMatrix;
		for (Y = 0; Y < B/subMatrixDim; Y++) {
      Y_offset = Y * B * subMatrixDim + threadIdx.y * B;
      for (X = 0; X < B/subMatrixDim; X++) {
      	X_offset = X * subMatrixDim + threadIdx.x;
        B1[Y_offset + X_offset] = X_GPU[chain_offset + Y_offset + X_offset];
      }
    }
		__syncthreads();
		
		for (Y = 0; Y < B/subMatrixDim; Y++) {
			Y_offset = Y * B * subMatrixDim + threadIdx.y * B;		
			for (X = 0; X < B/subMatrixDim; X++) {
				X_offset = X * subMatrixDim + threadIdx.x;
				result=0;
				__syncthreads();
				for (P=0; P < B; P++) {
					result += B0[Y_offset + P] * B1[X_offset + P * B];
				}			
				B2[X_offset + Y_offset] = result;
			}
		}
		
		__syncthreads();
		swapArray(B2, B0, B);
  }

  //X_GPU = B2;
  memcpy(R1_GPU, B2, sizeof(float)*B*B);

  return;
}

 			
