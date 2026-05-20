#define MSG_INIT	101   /* initialization values*/
#define MSG_OP		102   /* operator identifier  */
#define MSG_MATRIX_SIZE 201
#define MSG_MATRIX_VALUES 202

#ifndef MATRIX_H
	#define MATRIX_H
	#include "matrix.h"
#endif

matrix * broadcast_matrix (matrix *, int my_rank, int root_rank);
matrix * get_matrix (int source);
void send_matrix (matrix *, int destination);
void send_row (matrix *, int sent_row, int destination);
