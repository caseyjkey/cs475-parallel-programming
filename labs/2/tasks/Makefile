CC = gcc
OBJS = timer.o
FLAGS = -O3 -fopenmp
DEBUGFLAG = -DDEBUG
SEQFLAGS = -O3 -lm

EXEC = Merge_sortS Merge_sortP Merge_sortNP Merge_sort_SEQ Merge_sort_debug

all: $(EXEC)

Merge_sortS: Merge_sortS.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS)

Merge_sortP: Merge_sortP.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS)

Merge_sortNP: Merge_sortNP.c $(OBJS)
	$(CC) $(FLAGS)  -o $@ $< $(OBJS)

Merge_sort_SEQ: Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS)  -o $@ $< $(OBJS)

Merge_sort_debug:Merge_sort.c $(OBJS)
	$(CC) $(SEQFLAGS) $(DEBUGFLAG)  -o $@ $< $(OBJS)

timer.o: timer.c
	$(CC) -O3 -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)
