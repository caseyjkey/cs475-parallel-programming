EXEC = stencil_1D stencil_2D mat_vec stencil_1D_SEQ stencil_2D_SEQ  mat_vec_SEQ
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 

all: $(EXEC)

stencil_1D: stencil_1D.c $(OBJS) $(H_FILE)
		$(CC) $(FLAGS) -o $@ stencil_1D.c $(OBJS) $(MATHFLAG)

stencil_2D: stencil_1D.c $(OBJS) $(H_FILE)
		$(CC) $(FLAGS) -o $@ stencil_2D.c $(OBJS) $(MATHFLAG)

mat_vec: mat_vec.c $(OBJS) $(H_FILE)
		$(CC) $(FLAGS) -o $@ mat_vec.c $(OBJS) $(MATHFLAG)

stencil_1D_SEQ: stencil_1D.c $(OBJS) $(H_FILE)
		$(CC) $(SEQFLAGS) -o  $@ stencil_1D.c  $(OBJS)  $(MATHFLAG)

stencil_2D_SEQ: stencil_2D.c $(OBJS) $(H_FILE)
		$(CC) $(SEQFLAGS) -o  $@ stencil_2D.c  $(OBJS)  $(MATHFLAG)

mat_vec_SEQ: mat_vec.c $(OBJS) $(H_FILE)
		$(CC) $(SEQFLAGS) $(MATHFLAG) -o $@ mat_vec.c $(OBJS)


timer.o: timer.c
		$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean: 
		rm -f *.o $(OBJS) $(EXEC)
