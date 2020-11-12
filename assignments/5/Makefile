FLAGS= -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11 
CFLAGS=  -g -std=c99 -fopenmp  -I/usr/include/malloc/
LIBRARIES=-lm
CC=nvcc 
OBJS = MMScanCUDA.o MMScan.o 

all: plain verb int rand check CUDA CUDA-int

plain: $(OBJS)
	nvcc MMScan-wrapper.cu -o MMScan $(OBJS) $(FLAGS) -lm -DG=4 -DS=5


verb:
	$(CC) MMScan-wrapper.cu -o MMScan.verb $(OBJS) $(FLAGS) -DG=4 -DS=5


int: 
	$(CC) MMScan-wrapper.cu -o MMScan.int $(OBJS) $(FLAGS) -DG=4 -DS=5


rand: 
	$(CC) MMScan-wrapper.cu -o MMScan.rand $(OBJS) $(FLAGS) -DG=4 -DS=5



check: 
	$(CC) MMScan-wrapper.cu -o MMScan.check $(OBJS) $(FLAGS) -DCHECKING -DG=4 -DS=5


MMScan.o: MMScan.c
	gcc MMScan.c -o MMScan.o $(CFLAGS) $(LIBRARIES) -c

MMScanCUDA.o: MMScanKernel.cu
	$(CC) MMScanKernel.cu -o MMScanCUDA.o $(FLAGS) -c -DCUDA -DG=4 -DS=5
 

## Additional lines for DNC (& DNC.int for testing)

CUDA: MMScanCUDA.o
	$(CC) MMScan-wrapper.cu -o MMScanCUDA MMScanCUDA.o $(FLAGS) -DCUDA -DG=4 -DS=5

CUDA-int: 
	$(CC) MMScan-wrapper.cu -o MMScanCUDA.int MMScanCUDA.o $(FLAGS) -DINTERACTIVE -DG=4 -DS=5
 


clean:
	rm -f *.o MMScan MMScan.verb MMScan.int MMScan.rand MMScan.verb-rand MMScan.check MMScanCUDA MMScanCUDA.int 
