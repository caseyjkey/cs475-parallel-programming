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
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){ 
	long threadsPerBlock = (long) S;
  extern __shared__ float array[];
	
	// B0, B1 will hold intermediate products
	// B2 will hold results
  float* B0 = (float*)array;
  float* B1 = (float*)&array[B*B];
  float* B2 = (float*)&array[2*B*B];

  // Do calculation
	long X, Y, X_offset, Y_offset, P;
  long matricesPerBlock = N/gridDim.x;
	long matrixOffset = blockIdx.x * matricesPerBlock;
	long elementsPerMatrix = B * B;
	float result;

	memset(B0, 0, B * B * sizeof(float));
	for (Y = 0; Y < B; Y++) {
		B0[Y * B + Y] = 1;
	}

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
		swapArray(B2, B0, B);
  }

  memcpy(R1_GPU + (blockIdx.x * B * B), B0, sizeof(float)*B*B);

  return;
}

__global__ void MMScanKernel01(float* R1_GPU, float* R2_GPU, long N, long B){
  long threadsPerBlock = (long) S;
  long numMatrices = (long) G;
  extern __shared__ float array[];

  // B0, B1 will hold intermediate products
  // B2 will hold results
  float* B0 = (float*)array;
  float* B1 = (float*)&array[B*B];
  float* B2 = (float*)&array[2*B*B];

  // Do calculation
  long X, Y, X_offset, Y_offset, P;
  long elementsPerMatrix = B * B;
  float result;

  memset(B0, 0, elementsPerMatrix * sizeof(float));
  for (Y = 0; Y < B; Y++) {
    B0[Y * B + Y] = 1;
  }

  for (long Q = 0; Q < numMatrices; Q++) {
    long currentMatrix = Q * elementsPerMatrix;

    for (Y = 0; Y < B/threadsPerBlock; Y++) {
      Y_offset = Y * B * threadsPerBlock + threadIdx.y * B;

      for (X = 0; X < B/threadsPerBlock; X++) {
        X_offset = X * threadsPerBlock + threadIdx.x;

        B1[Y_offset + X_offset] = R1_GPU[Y_offset + X_offset];
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
    swapArray(B2, B0, B);
    memcpy(R2_GPU + (Q * elementsPerMatrix), B0, sizeof(float)*B*B);
  }

  return;
}	
