CFLAGS=  -O3  -std=c99 -fopenmp  -I/usr/include/malloc/ 
LIBRARIES=-lm
CC=gcc 
AR=xiar
OBJS = MMScan.o

all: plain verb int rand verb-rand check DNC DNC-int

debug: CFLAGS =-DDEBUG -g -Wall -Wextra -std=c99 -I/usr/include/malloc/
debug: all

plain: $(OBJS)
	$(CC) MMScan-wrapper.c -o MMScan $(OBJS) $(CFLAGS) $(LIBRARIES) 

verb:
	$(CC) MMScan-wrapper.c -o MMScan.verb $(OBJS) $(CFLAGS) $(LIBRARIES) -DVERBOSE

int: 
	$(CC) MMScan-wrapper.c -o MMScan.int $(OBJS) $(CFLAGS) $(LIBRARIES) -DINTERACTIVE

rand: 
	$(CC) MMScan-wrapper.c -o MMScan.rand $(OBJS) $(CFLAGS) $(LIBRARIES) -DRANDOM

verb-rand: 
	$(CC) MMScan-wrapper.c -o MMScan.verb-rand $(OBJS) $(CFLAGS) $(LIBRARIES) -DVERBOSE -DRANDOM

check: 
	$(CC) MMScan-wrapper.c -o MMScan.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DCHECKING

MMScan.o: MMScan.c
	$(CC) MMScan.c -o MMScan.o $(CFLAGS) $(LIBRARIES) -c

MMScanDNC.o: MMScan.c
	$(CC) MMScan.c -o MMScanDNC.o $(CFLAGS) $(LIBRARIES) -c -DDNC -DOTHER_FLAGS # (as needed)

## Additional lines for DNC (& DNC.int for testing)

DNC: MMScanDNC.o
	$(CC) MMScan-wrapper.c -o MMScanDNC MMScanDNC.o $(CFLAGS) $(LIBRARIES) -DDNC

DNC-int: 
	$(CC) MMScan-wrapper.c -o MMScanDNC.int MMScanDNC.o $(CFLAGS) $(LIBRARIES) -DINTERACTIVE


## Add additional line for different versions like verb, rand, etc. (for debugging/testing)

## Then add simolar sets of lines for other executables


clean:
	rm -f *.o MMScan MMScan.verb MMScan.int MMScan.rand MMScan.verb-rand MMScan.check MMScanDNC MMScanDNC.int 
