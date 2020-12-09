CC = gcc
EXEC = prog_SEQ prog_OMP
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 
N = 5

all: $(EXEC)
prog_SEQ: prog.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ prog.c  $(OBJS) -DSEQ

prog_OMP: prog.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ prog.c  $(OBJS) -DPAR

timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

check: prog_OMP prog_SEQ
	bash check.sh $(N)

clean: 
	rm -f *.o $(OBJS) $(EXEC)
