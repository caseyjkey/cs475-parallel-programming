FLAGS= -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11 
CC=nvcc 
OBJS = MMScan.o

all: plain verb int rand check CUDA CUDA-int

plain: $(OBJS)
	nvcc MMScan-wrapper.cu -o MMScan $(OBJS) $(FLAGS) 

verb:
	$(CC) MMScan-wrapper.cu -o MMScan.verb $(OBJS) $(FLAGS)

int: 
	$(CC) MMScan-wrapper.cu -o MMScan.int $(OBJS) $(FLAGS)

rand: 
	$(CC) MMScan-wrapper.cu -o MMScan.rand $(OBJS) $(FLAGS)


check: 
	$(CC) MMScan-wrapper.cu -o MMScan.check $(OBJS) $(FLAGS) -DCHECKING

MMScan.o: MMScan.cu
	$(CC) MMScan.cu -o MMScan.o $(FLAGS) -c

MMScanCUDA.o: MMScan.cu
	$(CC) MMScan.cu -o MMScanCUDA.o $(FLAGS) -c -DCUDA 

## Additional lines for DNC (& DNC.int for testing)

CUDA: MMScanCUDA.o
	$(CC) MMScan-wrapper.cu -o MMScanCUDA MMScanCUDA.o $(FLAGS) -DCUDA -DG=4 -DS=5

CUDA-int: 
	$(CC) MMScan-wrapper.cu -o MMScanCUDA.int MMScanCUDA.o $(FLAGS) -DINTERACTIVE -DG=4 -DS=5
 


clean:
	rm -f *.o MMScan MMScan.verb MMScan.int MMScan.rand MMScan.verb-rand MMScan.check MMScanCUDA MMScanCUDA.int 
