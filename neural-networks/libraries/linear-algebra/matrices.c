 /*
matrices.c file for matrices.h header file
Date: 02.09.2026
Author: Zubin Chowdhury
Purpose: Linear algebra library for performing matrix computations necessary in computer science and engineering.
*/

#include "matrices.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


mat *mat_init(size_t rows, size_t cols) {
    if (rows < 1 || cols < 1) {
        printf("Invalid matrix size.\n");
        return NULL;
    }

    size_t total_elements = (size_t) rows * (size_t) cols;
    double *array = calloc(total_elements, sizeof(double));
    if (!array) {
        printf("Memory allocation failed 1.\n");
        return NULL;
    }
    mat *mt = malloc(sizeof(mat));
    if (!mt) {
        printf("Failed memory allocation 2.\n");
        free(array);
        return NULL;
    }

    mt->rows = rows;
    mt->cols = cols;
    mt->arr = array;

    return mt;
}

mat *mat_copy(mat *mt) {
    mat *mt_new = mat_init(mt->rows, mt->cols);
    for (int i = 0, m = mt->rows, n = mt->cols; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mt_new->arr[i * n + j] = mt->arr[i * n + j];

        }
    }

    return mt_new;
}

mat *mat_user_fill(mat *mt) {
    for (int i = 0, m = mt->rows, n = mt->cols; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double tmp;
            printf("Input (%i, %i): ", i + 1, j + 1);
            scanf(" %lf", &tmp);
            mat_input(tmp, i, j, mt);
        }

    }
    return mt;
}

void mat_free(mat *mt) {
    free(mt->arr);
    free(mt);
    return;
}

void mat_print(mat *mt) {
    for (int i = 0, m = mt->rows, n = mt->cols; i < m; i++) {
        printf("[");
        for (int j = 0; j < n; j++) {
            printf("%0.2f", mat_select(mt, i, j));
            if (j != n - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

void mat_input(double value, size_t row, size_t col, mat *mt) {
    mt->arr[row * mt->cols + col] = value;
    return;
}

double mat_select(mat *mt, size_t row, size_t col) {
    return mt->arr[row * mt->cols + col];
}


mat *mat_add(mat *mtA, mat *mtB) {
    if (mtA->cols != mtB->cols || mtA->rows != mtB->rows) {
        printf("Invalid matrix operation.\n");
        return NULL;
    }

    mat *mt_result = mat_init(mtA->rows, mtA->cols);

    for (int i = 0, m = mtA->rows; i < m; i++) {
        for (int j = 0, n = mtA->cols; j < n; j++) {
            mat_input(mat_select(mtA, i, j) + mat_select(mtB, i, j), i, j, mt_result);
        }
    }

    return mt_result;
}

mat *mat_subtract(mat *mtA, mat *mtB) {
    if (mtA->cols != mtB->cols || mtA->rows != mtB->rows) {
        printf("Invalid matrix operation.\n");
        return NULL;
    }

    mat *mt_result = mat_init(mtA->rows, mtA->cols);

    for (int i = 0, m = mtA->rows; i < m; i++) {
        for (int j = 0, n = mtA->cols; j < n; j++) {
            mat_input(mat_select(mtA, i, j) - mat_select(mtB, i, j), i, j, mt_result);

        }
    }

    return mt_result;
}

mat *mat_value_mult(mat *mtA, mat *mtB) {
    if (mtA->cols != mtB->cols || mtA->rows != mtB->rows) {
        printf("Invalid matrix operation.\n");
        return NULL;
    }

    mat *mt_result = mat_init(mtA->rows, mtA->cols);

    for (int i = 0, m = mtA->rows; i < m; i++) {
        for (int j = 0, n = mtA->cols; j < n; j++) {
            mat_input(mat_select(mtA, i, j) * mat_select(mtB, i, j), i, j, mt_result);

        }
    }

    return mt_result;
}

mat *mat_scale(mat *mt, double scale_factor) {
    mat *mt_result = mat_init(mt->rows, mt->cols);

    for (int i = 0, m = mt->rows; i < m; i++) {
        for (int j = 0, n = mt->cols; j < n; j++) {
            mat_input(scale_factor * mat_select(mt, i, j), i, j, mt_result);

        }
    }

    return mt_result;
}

mat *mat_transpose(mat *mt) {
    mat *mt_transposed = mat_init(mt->cols, mt->rows);

    for (int i = 0, m = mt->rows, n = mt->cols; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat_input(mat_select(mt, i, j), j, i, mt_transposed);
        }
    }

    return mt_transposed;
}

mat *mat_mult(mat *mt1, mat *mt2) {
    if (mt1->cols != mt2->rows) {
        printf("Invalid matrix operation.\n");
        return NULL;
    }
    
    mat *mt_multiplied = mat_init(mt1->rows, mt2->cols);

    size_t m = mt1->rows;
    size_t n = mt2->cols;

    for (int i = 0; i < m; i++) {

        for (int j = 0; j < n; j++) {
            double value = 0;
            for (int k = 0; k < mt1->cols; k++) {
                value += mat_select(mt1, i, k) * mat_select(mt2, k, j);
            }
            mat_input(value, i, j, mt_multiplied);
        }
    }

    return mt_multiplied;
}

mat *mat_identity(size_t size) {
    mat *identity = mat_init(size, size);
    for (int i = 0; i < size; i++) {
        mat_input(1, i, i, identity);
    }
    
    return identity;
}

static double f_abs(double number) {
    uint64_t bits;
    memcpy(&bits, &number, sizeof(bits));

    bits &= 0x7FFFFFFFFFFFFFFFULL;

    memcpy(&number, &bits, sizeof(number));


    return number;
}

mat_LU *mat_LU_factorise(mat *mtA) {
    mat *mtL = mat_identity(mtA->rows);
    mat *mtU = mat_copy(mtA);
    mat *mtP = mat_identity(mtA->rows);

    int m = mtA->rows;
    int n = mtA->cols;

    for (int j = 0; j < m; j++) {
        double divisor = mat_select(mtU, j, j);
        
        if (f_abs(divisor) < 1e-12) {
            size_t swap_row = j;

            for (int k = j + 1; k < m; k++) {
                if (f_abs(mat_select(mtU, k, j)) > 1e-12) {
                    swap_row = k;
                    break;
                }
            }

            if (swap_row != j) {
                mat *mtP_new = mat_identity(m);

                for (int k = 0; k < n; k++) {
                    mat_input(0, swap_row, k, mtP_new);
                    mat_input(0, j, k, mtP_new);
                }

                mat_input(1, swap_row, j, mtP_new);
                mat_input(1, j, swap_row, mtP_new);

                mat *mtP_tmp = mtP;
                mtP = mat_mult(mtP_new, mtP);
                mat_free(mtP_tmp);

                mat *mtU_tmp = mtU;
                mtU = mat_mult(mtP_new, mtU);
                mat_free(mtU_tmp);
                mat_free(mtP_new);

                for (int col = 0; col < j; col++) {
                    double tmp = mat_select(mtL, j, col);
                    mat_input(mat_select(mtL, swap_row, col), j, col, mtL);
                    mat_input(tmp, swap_row, col, mtL);
                }

                divisor = mat_select(mtU, j, j);
            }
        }

        if (f_abs(divisor) < 1e-12) {
            continue; 
        }

        for (int i = j + 1; i < m; i++) {
            double scale_factor = mat_select(mtU, i, j) / divisor;
            mat_input(scale_factor, i, j, mtL);

            for (int k = 0; k < n; k++) {
                double value = mat_select(mtU, i, k) - (scale_factor * mat_select(mtU, j, k));
                mat_input(value, i, k, mtU);
            }
        }
    }

    mat_LU *factorised_matrices = malloc(sizeof(mat_LU));
    if (!factorised_matrices) {
        printf("Memory allocation failed for mat_LU.\n");
        mat_free(mtL);
        mat_free(mtU);
        mat_free(mtP);
        return NULL;
    }

    factorised_matrices->mtL = mtL;
    factorised_matrices->mtP = mtP;
    factorised_matrices->mtU = mtU;

    return factorised_matrices;
}


mat *mat_colA(mat *mt);
mat *mat_solve(mat *mtA, mat *vecB);


/* 
double *det(mat *mt);
mat *mat_inverse(mat *mt);
unsigned int *mat_rank(mat *mt);
mat *rref(mat *mt); 
mat *mat_vec_mult(mat *mt, mat *vec);
mat *mat_concatenate(mat *mt1, mat *mt2);
*/

// vector function prototypes
/*
mat *vec_init(size_t length) {
    


} */

size_t vec_size(mat *vec) {
    if (vec->rows != 1) {
        return vec->rows;
    }
    
    return vec->cols;
}

int confirmed_vec(mat *vec) {
    if (vec->cols == 1 || vec->rows == 1) {
        return 1;
    }

    printf("Invalid vector.\n");
    return 0;
}

double vec_dot(mat *vecA, mat *vecB) {
    if (!confirmed_vec(vecA) || !confirmed_vec(vecB)) return -1223.925;
    size_t size = vec_size(vecA);
    if (vec_size(vecB) != size) return -1223.925;

    // this function will take the dot product between two vectors regardless of whether they are column or row vectors
    double total = 0;
    for (int i = 0; i < size; i++) {
        total += vecA->arr[i] * vecB->arr[i];
    }

    return total;
}

mat *vec_cross(mat *vecA, mat *vecB) {
    size_t CROSS_PRODUCT_LENGTH = 3;
    if (
        !confirmed_vec(vecA) || !confirmed_vec(vecB) || 
        vec_size(vecA) != CROSS_PRODUCT_LENGTH || 
        vec_size(vecB) != CROSS_PRODUCT_LENGTH
    ) 
    return NULL;

    mat *vec_result = mat_init(CROSS_PRODUCT_LENGTH, 1);

    vec_result->arr[0] = vecA->arr[1] * vecB->arr[2] - vecA->arr[2] * vecB->arr[1];
    vec_result->arr[1] = vecA->arr[2] * vecB->arr[0] - vecA->arr[0] * vecB->arr[2];
    vec_result->arr[2] = vecA->arr[0] * vecB->arr[1] - vecA->arr[1] * vecB->arr[0];


    return vec_result;
}

/*
mat *vec_transpose(mat *vecA);
mat *vec_normal(mat *vec);
mat *vec_magnitude(mat *vec);
*/