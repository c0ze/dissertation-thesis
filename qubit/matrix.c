#include "matrix.h"
#include "standart.h"
#include <math.h>

matrix * create_matrix (int row_size, int column_size)
{
	matrix * m = (matrix *) malloc(sizeof(matrix));
	m->row_size = row_size;
	m->column_size = column_size;
	m->value = (__complex__ **) malloc (sizeof(__complex__ *)*row_size);
	int i;
	for(i=0; i<row_size; i++)
		m->value[i] = (__complex__ *)malloc(sizeof(__complex__)*column_size);
	return m;
}

matrix * create_qubit (int size, int value)
{
	if(value > power (2,size)) {
		error("error in create_qubit : impossible representation of in bits");
		return 0;
	}
	if (size <= 0) {
		error("error in create_qubit : first argument <= 0");
		return 0;
	}
	if (size == 1) {
	//printf("got values size %d  value %d\n",size, value);
	int left_over = value % 2;
	if (value >= 2) {
		error("error in create_qubit : value argument >= 2");
		return 0;
	}
	matrix * m = (matrix *) malloc(sizeof(matrix));
	m->row_size = 2;
	m->column_size = 1;
	m->value = (__complex__ **) malloc (sizeof(__complex__ *)*2);
	int i;
	for(i=0; i<2; i++) {
		m->value[i] = (__complex__ *)malloc(sizeof(__complex__));
		m->value[i][0] = (!left_over ^ i);
	}
	
	return m;
	
	}
	matrix * m1 = create_qubit(size-1,(value-(value >> size-1)*power(2,size-1)));
	matrix * m2 = create_qubit(1,(value >> size-1));
	matrix * m = tensor_product(m2, m1);
	
	//printf("matrix :\n");
	//print_matrix(m);
	//printf("\n");
	return m;
}

void init_matrix (matrix * m)
{
	int i, j;
	for(i=0; i < m->row_size; i++)
		for(j=0; j < m->column_size; j++)
			m->value[i][j] = 0;
}

void init_matrix_n (matrix * m, double n)
{
	int i, j;
	for(i=0; i < m->row_size; i++)
		for(j=0; j < m->column_size; j++)
			m->value[i][j] = n;
}

void init_matrix_c (matrix * m, __complex__ c)
{
	int i, j;
	for(i=0; i < m->row_size; i++)
		for(j=0; j < m->column_size; j++)
			m->value[i][j] = c;
}

void print_matrix (matrix * m)
{
	int i, j;
	for(i=0; i < m->row_size; i++) {
		for(j=0; j < m->column_size; j++)
			printf("(%.3f + %.3f i) ",__real__ m->value[i][j], __imag__ m->value[i][j]);
		printf("\n");
	}
}

void init_CNOT (matrix * m)
{
	if(m->column_size != 4 && m->row_size != 4)
		error("error in CNOT : Matrix size not good.");
	init_matrix(m);
	int i;
	for(i=0; i < m->row_size-2; i++)
			m->value[i][i] = 1;
	m->value[m->row_size-2][m->row_size-1] = 1;
	m->value[m->row_size-1][m->row_size-2] = 1;
}

void init_hadamard (matrix * m)
{
	if(m->row_size != 2 && m->column_size != 2)
		error("init hadamard - Wrong matrix size");
	init_matrix(m);
	int i;
	m->value[0][0] = 1 / sqrt(2);
	m->value[0][1] = 1 / sqrt(2);
	m->value[1][0] = 1 / sqrt(2);
	m->value[1][1] = -1 / sqrt(2);
}

matrix * tensor_product (matrix * a, matrix * b)
{
	int result_row_size = a->row_size*b->row_size;
	int result_col_size = a->column_size*b->column_size;
	matrix * result = create_matrix(result_row_size,result_col_size);

	init_matrix (result);
	int a_row,a_col,b_row,b_col;
	for(a_col=0; a_col < a->column_size; a_col++)
		for(a_row=0; a_row < a->row_size; a_row++)
			for(b_col=0; b_col < b->column_size; b_col++)
				for(b_row=0; b_row < b->row_size; b_row++) {
					//printf("assigning %e to %d  %d\n",(a->value[a_row][a_col]*b->value[b_row][b_col]),((b->row_size*a_row)+b_row),((b->column_size*a_col)+b_col));
					result->value[(b->row_size*a_row)+b_row][(b->column_size*a_col)+b_col] = Complex_Multiplication(a->value[a_row][a_col],b->value[b_row][b_col]);
				}
	
	return result;
}

matrix * dot_product (matrix * a, matrix * b)
{
	if(a->column_size != b->row_size) {
		print_matrix(a);
		print_matrix(b);
		error("dot product : Matrix mismatch.");
	}
	matrix * result = create_matrix(a->row_size,b->column_size);
	init_matrix(result);
	int row, col, i;
	__complex__ sum =0;
	for(row = 0; row < a->row_size; row++)
		for(col = 0; col< b->column_size; col++) {
			for(i=0; i < a->column_size; i++)
				sum += Complex_Multiplication(a->value[row][i], b->value[i][col]);
			result->value[row][col] = sum;
			sum = 0;
		}
	return result;
}
