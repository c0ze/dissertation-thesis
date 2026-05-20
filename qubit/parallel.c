#include "mpi.h"
#include "standart.h"

#ifndef PARALLEL_H
	#define PARALLEL_H
	#include "parallel.h"
#endif

#ifndef MATRIX_H
	#define MATRIX_H
	#include "matrix.h"
#endif

matrix * broadcast_matrix (matrix * m, int my_rank, int root_rank)
{
	int BroadCast_size [2];
	int i;
	
	if(my_rank == root_rank) 
	{
		BroadCast_size [0] = m->row_size;
		BroadCast_size [1] = m->column_size;
	}
	
	MPI_Bcast(BroadCast_size, 2, MPI_INT, root_rank, MPI_COMM_WORLD);
	
	double BroadCast_real_values [BroadCast_size[0] * BroadCast_size[1]];
	double BroadCast_imag_values [BroadCast_size[0] * BroadCast_size[1]];

	if(my_rank == root_rank) 
	{
		for(i=0; i < m->row_size * m->column_size; i++) {
			//printf("i : %d row s [%d] col s [%d] i/%d i %d",i, m->row_size,m->column_size,(i/m->column_size),(i%m->column_size));
			//printf("  value = %f %f i\n", __real__ m->value[i/m->column_size][i%m->column_size], __imag__ m->value[i/m->column_size][i%m->column_size]);
			BroadCast_real_values[i] = __real__ m->value[i/m->column_size][i%m->column_size];
			BroadCast_imag_values[i] = __imag__ m->value[i/m->column_size][i%m->column_size];
		}
	}
	
	MPI_Bcast(BroadCast_real_values, BroadCast_size[0] * BroadCast_size[1], MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Bcast(BroadCast_imag_values, BroadCast_size[0] * BroadCast_size[1], MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	
	
	m = create_matrix(BroadCast_size[0],BroadCast_size[1]);

	for (i=0; i < BroadCast_size[0]*BroadCast_size[1]; i++) {
		//printf("i : %d size 0 [%d] size 1 [%d] i/%d i %d",i,BroadCast_size[0],BroadCast_size[1],(i/BroadCast_size[1]),(i%BroadCast_size[0]));
		//printf("  value = %f %f i\n",BroadCast_real_values[i], BroadCast_imag_values[i]);
		__real__ m->value[i/BroadCast_size[1]][i%BroadCast_size[1]] = BroadCast_real_values[i];
		__imag__ m->value[i/BroadCast_size[1]][i%BroadCast_size[1]] = BroadCast_imag_values[i];
	}
	
	return m;
}

matrix * get_matrix (int source)
{
	MPI_Status status;
	int result [2];
	result[0] = 0;
	result[1] = 0;

	MPI_Recv(result, 2, MPI_INT, source, MSG_MATRIX_SIZE, MPI_COMM_WORLD, &status);
	
	//printf("received %d %d from %d.\n",result[0],result[1],source);
	
	double  real_values [result[1]];
	double  imag_values [result[1]];
	
	MPI_Recv(real_values, result[1], MPI_DOUBLE, source, MSG_MATRIX_VALUES, MPI_COMM_WORLD, &status);
	MPI_Recv(imag_values, result[1], MPI_DOUBLE, source, MSG_MATRIX_VALUES, MPI_COMM_WORLD, &status);
	
	matrix * m = create_matrix(result[1]/result[0],result[0]);
	int i;
	for (i=0; i < result[1]; i++) {
		//printf("i : %d size 0 [%d] size 1 [%d] i/%d i %d",i,result[0],result[1],(i/result[0]),(i%result[0]));
		//printf("  value = %f %f i\n",real_values[i], imag_values[i]);
		__real__ m->value[i/result[0]][i%result[0]] = real_values[i];
		__imag__ m->value[i/result[0]][i%result[0]] = imag_values[i];
	}
	
	//printf("received matrix \n");
	//print_matrix(m);
	return m;

}

void send_matrix (matrix * m, int destination)
{
	MPI_Request * request;
	
	int size [2];
	size [0] = m->column_size;
	size [1] = m->column_size * m->row_size;
	
	//printf("sending %d %d matrix to %d\n",size [0],size [1],destination);
	
	MPI_Send(size, 2, MPI_INT, destination, MSG_MATRIX_SIZE, MPI_COMM_WORLD);

	double real_values [m->row_size * m->column_size];
	double imag_values [m->row_size * m->column_size];
	
	int i;
	for(i=0; i < m->row_size * m->column_size; i++) {
		//printf("i : %d size 0 [%d] size 1 [%d] i/%d i %d",i,size[0],size[1],(i/m->column_size),(i%m->column_size));
		//printf("  value = %f %f i\n",m->value[i/m->column_size][i%m->column_size], m->value[i/m->column_size][i%m->column_size]);
		real_values[i] = __real__ m->value[i/m->column_size][i%m->column_size];
		imag_values[i] = __imag__ m->value[i/m->column_size][i%m->column_size];
	}
	MPI_Send(real_values, m->row_size * m->column_size, MPI_DOUBLE, destination, MSG_MATRIX_VALUES, MPI_COMM_WORLD);
	MPI_Send(imag_values, m->row_size * m->column_size, MPI_DOUBLE, destination, MSG_MATRIX_VALUES, MPI_COMM_WORLD);
	
}

void send_row (matrix * m, int sent_row, int destination)
{
	matrix * toBeSent = create_matrix(1,m->column_size);
	int i;
	//printf("root sending to %d:\n",destination);
	for(i = 0; i<m->column_size; i++)
		toBeSent->value[0][i]=m->value[sent_row][i];
	//print_matrix(toBeSent);
	send_matrix(toBeSent,destination);
	
	
}
