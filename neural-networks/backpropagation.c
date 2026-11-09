#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "libraries/linear-algebra/matrices.h"
#include "backpropagation.h"


// activation functions
mat *step(mat *z) {
    if (z->cols != 1) {
        printf("Invalid vector.\n");
        return NULL;
    }

    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;
    for (int i = 0, n = return_vec->rows; i < n; i++) {
        mat_input( (mat_select(z, i, 0) >= 0) ? 1 : 0, i, 0, return_vec); 
    }
    return return_vec;
}

mat *ReLU(mat *z) {
    if (z->cols != 1) {
        printf("Invalid vector.\n");
        return NULL;
    }

    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;
    for (int i = 0, n = return_vec->rows; i < n; i++) {
        mat_input( (mat_select(z, i, 0) >= 0) ? mat_select(z, i, 0) : 0, i, 0, return_vec); 
    }
    return return_vec;
}

mat *sigmoid(mat *z) {
    if (z->cols != 1) {
        printf("Invalid vector.\n");
        return NULL;
    }

    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;
    for (int i = 0, n = return_vec->rows; i < n; i++) {
        double selected_component = mat_select(z, i, 0);
        mat_input(1.0 / (1.0 + exp(-selected_component)), i, 0, return_vec); 
    }
    return return_vec;
}


