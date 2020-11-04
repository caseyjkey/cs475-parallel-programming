###
### CS475 Fall 17
### Makefile for CUDA PA4
### By Wim Bohm, and Waruna Ranasinghe
###

OPTIONS := -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11
EXECS := matmult00 vecMax00 vecMax01 matmult01 matmult10 matmult20 matmult30 matmult11 matmult21 matmult31

all:$(EXECS)

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)

matmultKernel00.o : matmultKernel.h matmultKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=16

matmultKernel10.o : matmultKernel.h matmultKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=8

matmultKernel20.o : matmultKernel.h matmultKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=4

matmultKernel30.o : matmultKernel.h matmultKernel00.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=2

matmultKernel01.o : matmultKernel.h matmultKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=16 -DFOOTPRINT_SIZE=32

matmultKernel11.o : matmultKernel.h matmultKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=8 -DFOOTPRINT_SIZE=32

matmultKernel21.o : matmultKernel.h matmultKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=4 -DFOOTPRINT_SIZE=32

matmultKernel31.o : matmultKernel.h matmultKernel01.cu 
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS) -DBLOCK_SIZE=2 -DFOOTPRINT_SIZE=32 

matmult00 : matmultKernel.h matmult.cu matmultKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=16

matmult10 : matmultKernel.h matmult.cu matmultKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=8 -DFOOTPRINT_SIZE=16

matmult20 : matmultKernel.h matmult.cu matmultKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=4 -DFOOTPRINT_SIZE=16

matmult30 : matmultKernel.h matmult.cu matmultKernel00.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=2 -DFOOTPRINT_SIZE=16


matmult01 : matmultKernel.h matmult.cu matmultKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=16 -DFOOTPRINT_SIZE=32

matmult11 : matmultKernel.h matmult.cu matmultKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=8 -DFOOTPRINT_SIZE=32

matmult21 : matmultKernel.h matmult.cu matmultKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=4 -DFOOTPRINT_SIZE=32

matmult31 : matmultKernel.h matmult.cu matmultKernel01.o timer.o
	nvcc $(filter-out $<,$^) -o $@ $(LIB) $(OPTIONS) -DBLOCK_SIZE=2 -DFOOTPRINT_SIZE=32

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

