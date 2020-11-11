###
### CS475 Fall 17
### Makefile for CUDA PA4
### By Wim Bohm, and Waruna Ranasinghe
###

OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11
EXECS := matmult00 vecMax00 vecMax01 matmult01 

all:$(EXECS)

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

matmultKernel00.o : matmultKernel.h matmultKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS)

matmultKernel01.o : matmultKernel.h matmultKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult00 : matmultKernel.h matmult.cu matmultKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) 

matmult01 : matmultKernel.h matmult.cu matmultKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DFOOTPRINT_SIZE=32

vecMaxKernel00.o : vecMaxKernel.h vecMaxKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS)

vecMaxKernel01.o : vecMaxKernel.h vecMaxKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS)


vecMax00 : vecMaxKernel.h vecMax.cu vecMaxKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS)

vecMax01 : vecMaxKernel.h vecMax.cu vecMaxKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS)


tar: matmultKernel00.cu matmultKernel01.cu matmultKernel.h matmult.cu Makefile vecMax.cu vecMaxKernel00.cu vecMaxKernel01.cu vecMaxKernel.h report.pdf
	tar cf PA5.tar $^

clean:
	rm -f $(EXECS) *.o

