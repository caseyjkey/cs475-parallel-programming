///
/// matmultKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;


  //int FOOTPRINT_SIZE = BLOCK_SIZE * 2;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue = 0;
  float Cvalue2 = 0;
  float Cvalue3 = 0;
  float Cvalue4 = 0;


  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Top left element
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    // Top right element
    shared_A[thread_row][thread_col + BLOCK_SIZE] = Asub[thread_row * A.stride + thread_col + BLOCK_SIZE];
    shared_B[thread_row][thread_col + BLOCK_SIZE] = Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];

    // Bottom left element
    shared_A[thread_row + BLOCK_SIZE][thread_col] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];
    shared_B[thread_row + BLOCK_SIZE][thread_col] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col];

    // Bottom right element
    shared_A[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = 
	Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col + BLOCK_SIZE];
    shared_B[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = 
	Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col + BLOCK_SIZE];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
    #pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e) {
       Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];
       Cvalue2 += shared_A[thread_row][e] * shared_B[e][thread_col + BLOCK_SIZE];
       Cvalue3 += shared_A[thread_row + BLOCK_SIZE][e] * shared_B[e][thread_col];
       Cvalue4 += shared_A[thread_row + BLOCK_SIZE][e] * shared_B[e][thread_col + BLOCK_SIZE];
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  Csub[thread_row * C.stride + thread_col] = Cvalue;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col] = Cvalue3;
  Csub[thread_row * C.stride + thread_col + BLOCK_SIZE] = Cvalue2;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col + BLOCK_SIZE] = Cvalue4;
}

