#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"          /* MPI constants and functions */
#include "standart.h"

#include <math.h>
	#ifndef MATRIX_H
	#define MATRIX_H
#include "matrix.h"
#endif

#ifndef PARALLEL_H
	#define PARALLEL_H
	#include "parallel.h"
#endif



#define NUM_SLAVES    4

int main(int argc, char** argv)
{
int size, rank;

MPI_Status status;

/* initlalize the MPI System     */
MPI_Init(&argc, &argv);

/* check for proper number of processes     */
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

matrix * input_vector;
matrix * quantum_program;

if (rank == 0)
{

	int qureg_size = 3;
	/* initialise the input vector*/

	input_vector = create_qubit(qureg_size,2);

	/* initialise the quantum program */

	quantum_program = create_matrix(2,2);
	init_hadamard(quantum_program);

	/* set the quantum program to an appropriate size */

	if (quantum_program->column_size < input_vector->row_size)
	{
		matrix * original_quantum_program = quantum_program;
		while (quantum_program->column_size < input_vector->row_size)
		{
			
			quantum_program = tensor_product(original_quantum_program,quantum_program);
		}
	}
	else if (quantum_program->column_size > input_vector->row_size)
	{
		error("Vector too small.");
		exit(0);
	}
	//print_matrix(quantum_program);
	//print_matrix(input_vector);
}
/* distribute the input vector */


input_vector = broadcast_matrix(input_vector,rank,0);

printf("input broadcast %d\n",rank);
//print_matrix(input_vector);


/* distribute the quantum program to slave processes */
int row_distributed ;
int size_2P;
if(rank == 0)
{
	size_2P = get_biggest_2s_power(size); /* in case the lam-mpi environment does not have a size that is a power of 2, we set a new size, and disregard the remaining nodes. */
	row_distributed = quantum_program->row_size / size_2P; /* number of rows distributed to each process */
	printf("row_dist %d\n",row_distributed);
}

MPI_Bcast (&row_distributed, 1, MPI_INT, 0, MPI_COMM_WORLD);
//printf("my rank : %d dist : %d\n",rank, row_distributed);

if(rank == 0) {
	int left_over = quantum_program->row_size % size_2P;
	int node_rank;
	int i = 0;
	for(node_rank = 1; node_rank < size_2P; node_rank++)
		for(i = node_rank*row_distributed; i < (node_rank+1)*row_distributed; i++)
			send_row(quantum_program, i, node_rank);

}

//MPI_Barrier(MPI_COMM_WORLD);

if(rank != 0)
{

int i;
matrix * program [row_distributed];
matrix * results [row_distributed];
for(i = 0; i < row_distributed; i++)
	program[i] = get_matrix(0);

for(i = 0; i < row_distributed; i++)
	results[i] = dot_product(program[i],input_vector);

printf("my rank : %d my result : \n",rank);
for(i = 0; i < row_distributed; i++)
	print_matrix(results[i]);

}

/* clean up and exit the MPI system         */
MPI_Finalize();

exit(EXIT_SUCCESS);
} /* and main() */    
