typedef struct {
	int row_size;
	int column_size;
	__complex__ double ** value;
} matrix;

matrix * create_matrix (int row_size, int column_size);
void init_matrix (matrix * m);
void print_matrix (matrix * m);
void init_CNOT (matrix * m);
void init_hadamard (matrix * m);
matrix * tensor_product (matrix * a, matrix * b);
matrix * dot_product (matrix * a, matrix * b);
matrix * create_qubit (int, int);
