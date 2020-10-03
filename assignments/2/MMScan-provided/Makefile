CFLAGS=  -g -std=c99 -fopenmp  -I/usr/include/malloc/ 
LIBRARIES=-lm
CC=gcc 
AR=xiar
OBJS = MMScan.o

all: plain verb int rand verb-rand check DNC DNC-int DNCP1 DNCP2 DNCP3 DNC-check DNC-verb DNCP1-check DNCP2-check DNCP3-check

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

DNCP1: MMScanDNCP1.o
	$(CC) MMScan-wrapper.c -o MMScanDNCP1 MMScanDNCP1.o $(CFLAGS) $(LIBRARIES) -DDNCP1

DNCP2: MMScanDNCP2.o
	$(CC) MMScan-wrapper.c -o MMScanDNCP2 MMScanDNCP2.o $(CFLAGS) $(LIBRARIES) -DDNCP2

DNCP3: MMScanDNCP3.o
	$(CC) MMScan-wrapper.c -o MMScanDNCP3 MMScanDNCP3.o $(CFLAGS) $(LIBRARIES) -DDNCP3


## Add additional line for different versions like verb, rand, etc. (for debugging/testing)

DNC-check: 
	$(CC) MMScan-wrapper.c -o MMScanDNC.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DDNC -DCHECKING

DNC-verb:
	$(CC) MMScan-wrapper.c -o MMScanDNC.verb $(OBJS) $(CFLAGS) $(LIBRARIES) -DDNC -DVERBOSE


## Then add simolar sets of lines for other executables

DNCP1-check:
	$(CC) MMScan-wrapper.c -o MMScanDNCP1.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DDNCP1 -DCHECKING

DNCP2-check:
	$(CC) MMScan-wrapper.c -o MMScanDNCP2.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DDNCP2 -DCHECKING

DNCP3-check:
	$(CC) MMScan-wrapper.c -o MMScanDNCP3.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DDNCP3 -DCHECKING


MMScanDNCP1.o: MMScan.c
	$(CC) MMScan.c -o MMScanDNCP1.o $(CFLAGS) $(LIBRARIES) -c -DDNCP1 -DOTHER_FLAGS # (as needed)

MMScanDNCP2.o: MMScan.c
	$(CC) MMScan.c -o MMScanDNCP2.o $(CFLAGS) $(LIBRARIES) -c -DDNCP2 -DOTHER_FLAGS # (as needed)

MMScanDNCP3.o: MMScan.c
	$(CC) MMScan.c -o MMScanDNCP3.o $(CFLAGS) $(LIBRARIES) -c -DDNCP3 -DOTHER_FLAGS # (as needed)

clean:
	rm -f *.o MMScan MMScan.verb MMScan.int MMScan.rand MMScan.verb-rand MMScan.check MMScanDNC MMScanDNC.int MMScanDNC.verb MMScanDNC.check MMScanDNCP1.check MMScanDNCP2.check MMScanDNCP2.check MMScanDNCP1 MMScanDNCP2 MMScanDNCP3 
