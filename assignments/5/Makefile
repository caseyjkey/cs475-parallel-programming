###
### CS475 Fall 17
### Makefile for CUDA PA4
### By Wim Bohm, and Waruna Ranasinghe
###

OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11
EXECS := MMScan MMScanCUDA 

all:$(EXECS)

MMScanKernel.o : MMScanKernel.h MMScanKernel.cu 
	nvcc -g -G $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DG=3 -DS=3

MMScan : MMScanKernel.h MMScan-wrapper.cu MMScanKernel.o
	nvcc -g -G $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DG=3 -DS=3

MMScanCUDA : MMScanKernel.h MMScan-wrapper.cu MMScanKernel.o
	nvcc -g -G $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DCUDA -DG=3 -DS=3


tar: MMScanKernel.cu MMScan-wrapper.cu MMScanKernel.h Makefile report.pdf
	tar cf PA5.tar $^

clean:
	rm -f $(EXECS) *.o

