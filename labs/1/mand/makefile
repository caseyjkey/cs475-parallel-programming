CC = gcc
EXEC = mandSEQ mandOMP
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3

all: $(EXEC)

mandSEQ: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ mandelbrot.c $(OBJS)

mandOMP: mandelbrot.c $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ mandelbrot.c $(OBJS)


timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
	rm -f *.o *.pgm $(OBJS) $(EXEC)
