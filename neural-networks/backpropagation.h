#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <stddef.h>
#include "libraries/linear-algebra/matrices.h"

#define LAYERS_NUMBER 5

// structs

typedef struct LAYER {
    double *neuron;
    mat *bias_vec;
    mat *weights;
    const unsigned int NUMBER_OF_NEURONS;
} LAYER;

typedef struct network {
    LAYER *layer;
    const unsigned int NUMBER_OF_LAYERS;
} network;

// prototypes

// activation functions
mat *step(mat *z);
mat *ReLU(mat *z);
mat *sigmoid(mat *z);



#endif