/// matmultKernel.h
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-19 DVN
///
/// Kernels defined with this header must 
/// multiply two matrices using CUDA: A x B = C
///

#ifndef __MMKERNEL__
#define __MMKERNEL__

// Defines for size of thread block and data computed by a thread block
#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.


typedef struct {
  int width;
  int height;
  int stride;
  float* elements;
} Matrix;

// Forward declaration of the kernel function that performs the work.
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#endif

