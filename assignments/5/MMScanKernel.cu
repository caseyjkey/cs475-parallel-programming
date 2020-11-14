#include "MMScanKernel.h"

// Find N/G breakpoints for our scan
// G groups/blocks
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){ 
	long threadsPerBlock = (long) S;
	
	// B0, B1 will hold intermediate products
	// B2 will hold results

  extern __shared__ float array[];
  float* B0 = (float*)array;
  float* B1 = (float*)&array[B*B];
  float* B2 = (float*)&array[2*B*B];
	float* Btemp;

  // Do calculation
	long X, Y, X_offset, Y_offset, P;
  long matricesPerBlock = N/gridDim.x;
	long matrixOffset = blockIdx.x * matricesPerBlock;
	long elementsPerMatrix = B * B;
	long result;

	memset(B0, 0, B * B * sizeof(float));
	__syncthreads();
	for (Y = 0; Y < B; Y++) {
		B0[Y * B + Y] = 1;
	}
	__syncthreads();

  for (long Q = 0; Q < matricesPerBlock; Q++) {
    long currentMatrix = Q * elementsPerMatrix;
    long chain_offset = matrixOffset * elementsPerMatrix + currentMatrix;

		for (Y = 0; Y < B/threadsPerBlock; Y++) {
      Y_offset = Y * B * threadsPerBlock + threadIdx.y * B;

      for (X = 0; X < B/threadsPerBlock; X++) {
      	X_offset = X * threadsPerBlock + threadIdx.x;

        B1[Y_offset + X_offset] = X_GPU[chain_offset + Y_offset + X_offset];
      }
    }
		
		for (Y = 0; Y < B/threadsPerBlock; Y++) {
			Y_offset = Y * B * threadsPerBlock + threadIdx.y * B;		

			for (X = 0; X < B/threadsPerBlock; X++) {
				X_offset = X * threadsPerBlock + threadIdx.x;
				result=0;

				for (P=0; P < B; P++) {
					result += B0[Y_offset + P] * B1[X_offset + P * B];
				}			

				B2[X_offset + Y_offset] = result;
			}
		}

		__syncthreads();
		Btemp = B0;
		B0 = B2;
		B2 = Btemp;
		
  }

  memcpy(R1_GPU + (sizeof(float) * (blockIdx.x * B * B)), B0, sizeof(float)*(B*B));

  return;
}
