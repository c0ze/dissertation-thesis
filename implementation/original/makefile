all : qubit

qubit : qubit.c matrix.c standart.c parallel.c
	mpicc qubit.c matrix.c standart.c parallel.c -lm -o qubit
