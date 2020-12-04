OBJS = timer.o
H_FILE = timer.h

EXEC = jac

all: $(EXEC)

jac: jac.c $(OBJS) $(H_FILE)
	gcc -O3 -o $@ jac.c $(OBJS) $(MATHFLAG)

clean:
	rm -f $(EXEC)

tar:
	tar cvf PA4.tar Makefile jacOMP.c jacMPI.c report.pdf
