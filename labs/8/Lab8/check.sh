#!/bin/bash

if [[ -z "$1" ]]; then
	N=5
else
	N=$1
fi

cmd="diff <(./prog_SEQ $N 0 | grep -v elapsed) <(./prog_OMP $N 0 | grep -v elapsed)"

if [[ -z "$(eval $cmd)" ]]; then
	echo "Success, no errors."
else
	echo "There are errors:"
	echo $cmd
	eval $cmd
fi
