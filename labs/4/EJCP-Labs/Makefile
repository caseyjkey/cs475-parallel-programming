OBJS := syr2k syr2k-perm syr2kP syr2k-permP

all: $(OBJS)

syr2k: syr2k.c
	gcc -O3 -o $@ $<

syr2k-perm: syr2k-perm.c
	gcc -O3 -o $@ $<

syr2kP: syr2k.c
	gcc -fopenmp -O3 -o $@ $<

syr2k-permP: syr2k-perm.c
	gcc -fopenmp -O3 -o $@ $<


clean:
	rm -rf $(OBJS)
