// matrices.h
#ifndef MATRICES_H
#define MATRICES_H

#include <stddef.h>

// data structures

typedef struct mat {
    double *arr;
    size_t rows;
    size_t cols;
} mat;


typedef struct mat_LU {
    mat *mtL;
    mat *mtU;
    mat *mtP;
} mat_LU;



// function prototypes

double mat_select(mat *mt, size_t row, size_t col);
void mat_input(double value, size_t row, size_t col, mat *mt);

// maths functions
static double f_abs(double number);

// memory management and data visualisation
mat *mat_copy(mat *mt);
mat *mat_init(size_t rows, size_t cols);
mat *mat_user_fill(mat *mt);
void mat_free(mat *mt);
void mat_print(mat *mt);

// matrix operations
mat *mat_add(mat *mtA, mat *mtB); // adds two matrices together
mat *mat_identity(size_t size);  // constructs a square identity matrix of a desired n x n size
mat *mat_mult(mat *mt1, mat *mt2); // multiplies two matrices together.
mat *mat_scale(mat *mt, double scale_factor); // multiplies a matrix by a scalar.
mat *mat_subtract(mat *mtA, mat *mtB); // substracts two matrices.
mat *mat_transpose(mat *mt); // transposes a matrix.
mat *mat_value_mult(mat *mtA, mat *mtB); // calculates a value by value multiplication of two matrices.

mat_LU *mat_LU_factorise(mat *mtA); // performs A = LU decomposition and returns both matrices L and U.
mat *mat_colA(mat *mt);
mat *mat_solve(mat *mtA, mat *vecB);


double *det(mat *mt);
mat *mat_inverse(mat *mt);
unsigned int *mat_rank(mat *mt);
mat *rref(mat *mt); 
mat *mat_vec_mult(mat *mt, mat *vec);
mat *mat_concatenate(mat *mt1, mat *mt2);

// vector function prototypes
mat *vec_init(size_t length);
size_t vec_size(mat *vec);
int confirmed_vec(mat *vec);
double vec_dot(mat *vecA, mat *vecB);
mat *vec_cross(mat *vecA, mat *vecB);
mat *vec_transpose(mat *vecA);
mat *vec_normal(mat *vec);
mat *vec_magnitude(mat *vec);



#endif