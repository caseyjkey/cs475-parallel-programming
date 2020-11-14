#include "MMScanKernel.h"

// Find N/G breakpoints for our scan
// G groups/blocks
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){ 
  extern __shared__ float array[];
	
	// B0, B1 will hold intermediate products
	// B2 will hold results
  float* B0 = array;
  float* B1 = &array[B*B];
  float* B2 = &array[2*B*B];
	float* Btemp;

  // Do calculation
	long X, Y, X_offset, Y_offset, P;
  long matricesPerBlock = N/gridDim.x;
	long matrixOffset = blockIdx.x * matricesPerBlock;
	long elementsPerMatrix = B * B;
	long result;

	memset(B0, 0, B * B * sizeof(float));
	B0[0] = 1;
  B1[0] = 1;
	B2[0] = 1;
	for (long i = 0; i < B; i++)
		B0[i * B + i] = 1;
	__syncthreads();

  for (long Q = 0; Q < matricesPerBlock; Q++) {
    long currentMatrix = Q * elementsPerMatrix;
    long chain_offset = matrixOffset * elementsPerMatrix + currentMatrix;
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
				X_offset = X * S + threadIdx.x;
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
