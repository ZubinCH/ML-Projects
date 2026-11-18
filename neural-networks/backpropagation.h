#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <stddef.h>
#include "libraries/linear-algebra/matrices.h"

#define MAX_LINE 1 * 1024 * 1024
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
void load_mat_array_with_file(size_t NUMBER_OF_MATRICES, mat **matrices, FILE *file);
void network_free(LAYER *layer, size_t count);

// Function performs supervised learning on a neural network.
bool train(
    const size_t NUMBER_OF_DATA, 
    const size_t NUMBER_OF_LAYERS,
    const size_t *NUMBER_OF_NEURONS_IN_LAYER,
    const char *activation_function_choice,
    const char *input_data, 
    const char *expected_outputs, 
    const char *output_weights,
    const char *output_bias
    );



#endif