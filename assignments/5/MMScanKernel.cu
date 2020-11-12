#include "matmultKernel.h"

__global__ void MMScanKernel(float*** X, float*** Y, float*** T, long start, long end, long size);
