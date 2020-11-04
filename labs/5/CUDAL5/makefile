###
### CS475 Fall 17
### Makefile for CUDA L8
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
###

OPTIONS   :=  -O3 --ptxas-options -v --gpu-architecture=sm_61 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11 -I/s/bach/g/under/caseykey/cuda-patches/include

EXECS :=  vecadd00 vecadd01
all:$(EXECS)

clean:
	rm -f $(EXECS) *.o

timer.o : timer.cu timer.h
	nvcc $< -c -o $@ $(OPTIONS)


vecaddKernel00.o : vecaddKernel00.cu
	nvcc $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	nvcc $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

vecaddKernel01.o : vecaddKernel01.cu
	nvcc $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	nvcc $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


