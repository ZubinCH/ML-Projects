#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "libraries/linear-algebra/matrices.h"
#include "backpropagation.h"

// activation functions
mat *step(mat *z) {
    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;

    for (int i = 0, n = return_vec->rows; i < n; i++) {
        mat_input((mat_select(z, i, 0) >= 0) ? 1 : 0, i, 0, return_vec); 
    }
    return return_vec;
}

mat *ReLU(mat *z) {
    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;

    for (int i = 0, n = return_vec->rows; i < n; i++) {
        mat_input( (mat_select(z, i, 0) >= 0) ? mat_select(z, i, 0) : 0, i, 0, return_vec); 
    }
    return return_vec;
}

mat *sigmoid(mat *z) {
    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;

    for (int i = 0, n = return_vec->rows; i < n; i++) {
        double selected_component = mat_select(z, i, 0);
        mat_input(1.0 / (1.0 + exp(-selected_component)), i, 0, return_vec); 
    }
    return return_vec;
}

mat *sigmoid_derivative(mat *z) {
    mat *return_vec = mat_init(z->rows, 1);
    if (!return_vec) return NULL;

    for (int i = 0, n = return_vec->rows; i < n; i++) {
        double selected_component = mat_select(z, i, 0);
        mat_input(exp(-selected_component) / ((1.0 + exp(-selected_component)) * (1.0 + exp(-selected_component))), i, 0, return_vec); 
    }
    return return_vec;
}

mat *matZ(mat *inputs, LAYER *layer) {
    if (inputs->rows == 1) {
        mat *tmp1 = inputs;
        inputs = mat_transpose(inputs);
        mat_free(tmp1);
    }

    else if (inputs->cols != 1) {
        printf("Invalid inputs vector.\n");
        return NULL;
    }

    mat *Wx = mat_mult(layer->weights, inputs);
    mat *z = mat_add(Wx, layer->bias);
    mat_free(Wx);

    return z;
}


mat *activation_function(mat *z, char activation) {

    switch (activation) {
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

mat *activation_function_derivative(mat *z, LAYER *layer) {
    switch (layer->actvnfunc) {
        case 's':
            return mat_init(z->rows, z->cols);
        case 'o':
            return sigmoid_derivative(z);
        case 'r':
            return step(z);
        default:
            printf("Invalid activation function.\n");
            return NULL;
    }
}

void load_matrix_with_file(mat *mt, FILE *file) {
    char line[MAX_LINE];
    size_t m_max = mt->rows;
    size_t n_max = mt->cols;

    size_t m = 0;
    while(fgets(line, sizeof(line), file) && m < m_max) {
        line[strcspn(line, '\n')] = '\0';
        char *token = strtok(line, ",");
        if (!token) {
            printf("Failed to tokenise.\n");
            return;
        }

        size_t n = 0;
        while (token != NULL && n < n_max) {
            double value = atof(token);
            mat_input(value, m, n, mt);
            token = strtok(NULL, ",");
            n++;
        }
        m++;
    }
}

void train(
    const size_t NUMBER_OF_INPUT_NEURONS, 
    const size_t NUMBER_OF_DATA, 
    const size_t NUMBER_OF_EXPECTED_OUTPUTS, 
    const size_t NUMBER_OF_LAYERS,
    const size_t *NUMBER_OF_NEURONS_PER_LAYER,
    const char *input_data, 
    const char *expected_outputs, 
    const char *output_weights,
    const char *output_bias,
    const char activation_function_choice
    ) 
    {
        // Open files
        FILE *input_file = fopen(input_data, 'r');
        if (!input_file) {
            printf("Failed to open %s.\n", input_data);
            return;
        }

        FILE *expected_outputs_file = fopen(expected_outputs, 'r');
        if (!expected_outputs_file) {
            printf("Failed to open %s.\n", expected_outputs);
            return;
        }

        FILE *output_weights_file = fopen(output_weights, 'w');
        if (!output_weights_file) {
            printf("Failed to open %s.\n", output_weights);
            return;
        }

        FILE *output_bias_file = fopen(output_bias, 'w');
        if (!output_bias_file) {
            printf("Failed to open %s.\n", output_bias);
            return;
        }

        // Tokenise inputs and store into matrix
        mat *input_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_INPUT_NEURONS);
        if (!input_matrix) return;
        load_matrix_with_file(input_matrix, input_file);

        // Tokenise expected outputs and store into matrix
        mat *expected_outputs_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_EXPECTED_OUTPUTS);
        if (!expected_outputs_matrix) return;
        load_matrix_with_file(expected_outputs_matrix, expected_outputs_file);


        LAYER *layer = malloc(NUMBER_OF_LAYERS * sizeof(LAYER));







        mat ***weight_matrix_derivatives = (mat ***) malloc(NUMBER_OF_DATA * sizeof(mat **));
        mat ***bias_vector_derivatives = (mat ***) malloc(NUMBER_OF_DATA * sizeof(mat **));

        for (int i = 0; i < NUMBER_OF_DATA; i++) {
            weight_matrix_derivatives[i] = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
            bias_vector_derivatives[i] = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
            for (int j = 0; j < NUMBER_OF_LAYERS; j++) {
                weight_matrix_derivatives[i][j] = mat_init(thinker->layer[j]->weights->rows, thinker->layer[j]->weights->cols);
                bias_vector_derivatives[i][j] = mat_init(thinker->layer[j]->NUMBER_OF_NEURONS, 1);
            }
        }




        fclose(input_file);
        fclose(expected_outputs_file);
        fclose(output_weights_file);
        fclose(output_bias_file);
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
        returned biases (write)
    End open files
        
    Initialise the inputs matrix
        tokenise inputs
        - each input layer is a row in the inputs matrix (data x inputs)      
        
    Initialise expected outputs matrix
        tokenise expected outputs
        - each expected output is a row in the inputs matrix
        - an nth row in the output matrix represents the expected output for an nth row in the inputs matrix
    
    Initialise network struct
        store the number of layers
        allocate memory for each layer
            store the number of neurons per layer
            allocate memory for neuron vector per layer
            allocate memory for bias vector between two layers
            allocate memory for weight matrix between two layers
    
    Initialise derivative matrices    // cost function's gradient vector
        mat ***weight_matrix_derivatives = malloc(number_of_data * sizeof(mat**));
        mat ***bias_vector_derivatives = malloc(number_of_data * sizeof(mat**));
        for (int i = 0; i < number_of_data; i++) {
            weight_matrix_derivatives[i] = (mat**) malloc(number_of_layers * sizeof(mat**));
            bias_vector_derivatives[i] = (mat**) malloc(number_of_layers * sizeof(mat**));
            for (int j = 0; j < number_of_layers; j++) {
                weight_matrix_derivatives[i][j] = mat_init(network->layer[j]->weights->rows, network->layer[j]->weights->cols);
                bias_vector_derivatives[i][j] = mat_init(network->layer[j]->NUMBER_OF_NEURONS, 1);
            }
        }
        - each column will store the results in gradient calculations per parameter per training data
        const total_number_of_parameters = 0
        FOR (i = 1, i < NUMBER_OF_LAYERS, i++) 
            total_number_of_parameters += (layer[i]->number_of_neurons + 1) * layer[i - 1]->number_of_neurons; // number of weights + number of biases
        END FOR
        del_C = mat_init(total_number_of_parameters, number of data);


    
    END Initialisation

    // 1. loop through training data
    FOR (i = 0, i < number_of_data, i++)
        mat *inputs_vector = mat_select_row(


    END FOR Loop


    // 2. Calculate del_C


    // 3. Apply gradient descent
        

            
     





        Return network 
*/
