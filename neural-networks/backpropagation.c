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
        mat_input((mat_select(z, i, 0) >= 0) ? 1 : 0, i, 0, return_vec); 
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

mat *activation_function(mat *inputs, LAYER *layer) {
    if (inputs->rows == 1) {
        mat *tmp = inputs;
        inputs = mat_transpose(inputs);
        free(tmp);
    }

    else if (inputs->cols != 1) {
        printf("Invalid inputs vector.\n");
        return NULL;
    }

    mat *z = mat_add(mat_mult(layer->weights, inputs), layer->bias);

    switch (layer->ACTIVATION_FUNCTION) {
        case 's':
            return step(z);
        case 'o':
            return sigmoid(z);
        case 'r':
            return ReLU(z);
        default:
            printf("Invalid activation function.\n");
            return NULL;
    }

}

mat *mat_z(LAYER layer, double *input) { 


}


/*
    Training with backpropagation pseudocode:
    
    Inputs: 
        name of file with inputs
        number of input neurons
        number of data
        name of file with expected outputs
        number of expected outputs
        number of layers
    END Inputs

    Open files
        inputs (read)
        expected outputs (read)
        returned weights (write)
    End open files
        
    Initialise the inputs matrix
        tokenise inputs
        -each input layer is a row in the inputs matrix (data x inputs)      
        
    Initialise expected outputs matrix
        tokenise expected outputs
        -each expected output is a row in the inputs matrix
        -an nth row in the output matrix represents the expected output for an nth row in the inputs matrix
    
    Initialise network struct
        store the number of layers
        allocate memory for each layer
            store the number of neurons per layer
            allocate memory for neuron vector per layer
            allocate memory for bias vector between two layers
            allocate memory for weight matrix between two layers

        

            
        Open     
        Initialise network struct





        Return network 
*/
