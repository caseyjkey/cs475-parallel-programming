#!/bin/bash
# This is a comment!
echo Calling MMScan.verify-rand repeatedly
for N in {20..500..10}
do
    for B in {4..24..4}
    do
	echo
	echo "                    " N = $N and B = $B
	for j in {1..4}
	do
	    ./MMScan.verify-rand $N $B
	done
    done
done
