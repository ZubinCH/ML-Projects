#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <stddef.h>
#include "libraries/linear-algebra/matrices.h"

#define MAX_LINE 2048
#define LEARNING_RATE 0.0001

// structs

typedef struct LAYER {
    mat *neuron;
    mat *bias;
    mat *weights;
    size_t neuron_count;
    char actvnfunc;
} LAYER;

// prototypes

// activation functions
mat *step(mat *z);
mat *ReLU(mat *z);
mat *sigmoid(mat *z);
mat *sigmoid_derivative(mat *z);



#endif