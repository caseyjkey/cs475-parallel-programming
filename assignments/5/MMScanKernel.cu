#include "MMScanKernel.h"

// Find N/G breakpoints for our scan
// G groups/blocks
__global__ void MMScanKernel00(float* X_GPU, float* R1_GPU, long N, long B){
	long matrixSize = sizeof(float)*B*B;
	extern __shared__ float array[];
	
	float* B1 = array;
	float* B2 = &array[B*B];
	float* B3 = &array[2*B*B];

	// Do calculation
		
	long chain_offset = blockIdx.x * B * B;
	for (long Y = 0; Y < B/S; Y++) {
		long Y_offset = Y * B * S + threadIdx.y * B;
		for (long X = 0; X < B/S; X++) {
			long X_offset = X * S + threadIdx.x;
			 B1[Y_offset + X_offset] = X_GPU[chain_offset + Y_offset + X_offset];
		}
	}
	// Save result to device memory
	__syncthreads();

	X_GPU[0] = G;
	
	return;
}


