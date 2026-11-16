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
    char actvnfunc;
} LAYER;

// prototypes

// activation functions
double activation_function (double z, char function_choice);
double activation_function_derivative(double z, char function_choice);
void mat_apply_activation_function(mat *z, char function_choice);
void mat_apply_activation_function_derivative(mat *z, char function_choice);

mat *matZ(mat *inputs, LAYER *layer);

void initial_weights_selection(mat *weights, double mean, double stdv);

void load_matrix_with_file(mat *mt, FILE *file);
void network_free(LAYER *layer, size_t count);



#endif