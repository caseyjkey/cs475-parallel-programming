OBJS = timer.o
H_FILE = timer.h

EXEC = jac jacMPI jacOMP

all: $(EXEC)

jac: jac.c $(OBJS) $(H_FILE)
	gcc -O3 -o $@ jac.c $(OBJS) $(MATHFLAG)

jacOMP: jacOMP.c $(OBJS) $(H_FILE)
	gcc -O3 -fopenmp -o $@ jacOMP.c $(OBJS) $(MATHFLAG)

jacMPI: jacMPI.c $(OBJS) $(H_FILE)
	mpicc -O3 -o $@ jacMPI.c $(OBJS) $(MATHFLAG)

clean:
	rm -f $(EXEC)

tar:
	tar cvf PA4.tar Makefile jacOMP.c jacMPI.c report.pdf
