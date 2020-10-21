CC = gcc
OBJS = timer.o
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 
MATHFLAGS = -lm
EXEC = sieve sieve1 sieve2 sieve3 sieve4 sieve4-1

all: $(EXEC)

sieve: sieve.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

sieve1: sieve1.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

sieve2: sieve2.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

sieve3: sieve3.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

sieve4: sieve4.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

sieve4-1: sieve4-1.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS) $(MATHFLAGS)

timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)

PA3.tar: sieve.c sieve1.c sieve2.c sieve3.c sieve4.c report.pdf makefile
	tar cvf PA3.tar sieve.c sieve1.c sieve2.c sieve3.c sieve4.c report.pdf makefile
