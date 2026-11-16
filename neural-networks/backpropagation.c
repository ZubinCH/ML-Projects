#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "backpropagation.h"
#include "libraries/linear-algebra/matrices.h"
#include "libraries/random-numbers/random_numbers.h"

// activation functions
mat *step(mat *z) {
    size_t rows = z->rows;
    size_t cols = z->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (!(mat_select(z, i, j) > 0)) mat_input(0, i, j, z);
            else mat_input(1, i, j, z);
        }
    }

    return z;
}

mat *ReLU(mat *z) {
    size_t rows = z->rows;
    size_t cols = z->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (!(mat_select(z, i, j) > 0)) mat_input(0, i, j, z);
        }
    }

    return z;
}

mat *sigmoid(mat *z) {
    size_t rows = z->rows;
    size_t cols = z->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double selected_component = mat_select(z, i, j);
            double sigmoid_value = 1.0 / (1.0 + exp(-selected_component));
            mat_input(sigmoid_value, i, j, z);
        }
    }

    return z;
}

mat *sigmoid_derivative(mat *z) {
    size_t rows = z->rows;
    size_t cols = z->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double selected_component = mat_select(z, i, j);
            double sigmoid_derivative_value = exp(-selected_component) / ((1.0 + exp(-selected_component)) * (1.0 + exp(-selected_component)));
            mat_input(sigmoid_derivative_value, i, j, z);
        }
    }

    return z;
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
            mat *new_z = mat_init(z->rows, z->cols);
            mat_free(z);
            return new_z;
        case 'o':
            return sigmoid_derivative(z);
        case 'r':
            return step(z);
        default:
            printf("Invalid activation function.\n");
            return NULL;
    }
}

void initial_weights_selection(mat *weights, double mean, double stdv) {
    size_t rows = weights->rows;
    size_t cols = weights->cols;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double value = normally_distributed_random_number(mean, stdv);
            mat_input(value, i, j, weights);
        }
    }

    return;
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

bool train(
    const size_t NUMBER_OF_DATA, 
    const size_t NUMBER_OF_LAYERS,
    const size_t *NUMBER_OF_NEURONS_IN_LAYER,
    const char *activation_function_choice,
    const char *input_data, 
    const char *expected_outputs, 
    const char *output_weights,
    const char *output_bias
    ) 
    {
        srand((unsigned int) time(NULL)); // set new seed for initialising matrices

        // Initialise constants
        const size_t NUMBER_OF_EXPECTED_OUTPUTS = NUMBER_OF_NEURONS_IN_LAYER[NUMBER_OF_LAYERS - 1];
        const size_t NUMBER_OF_INPUT_NEURONS = NUMBER_OF_NEURONS_IN_LAYER[0];

        // Open files

        // Load inputs, tokenise and store into matrix
        FILE *input_file = fopen(input_data, 'r');
        if (!input_file) {
            printf("Failed to open %s.\n", input_data);
            return false;
        }
        mat *input_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_NEURONS_IN_LAYER[0]);
        if (!input_matrix) return false;
        load_matrix_with_file(input_matrix, input_file);
        fclose(input_file);

        // Load expected outputs, tokenise and store into matrix
        FILE *expected_outputs_file = fopen(expected_outputs, 'r');
        if (!expected_outputs_file) {
            printf("Failed to open %s.\n", expected_outputs);
            return false;
        }
        mat *expected_outputs_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_EXPECTED_OUTPUTS);
        if (!expected_outputs_matrix) return false;
        load_matrix_with_file(expected_outputs_matrix, expected_outputs_file);
        fclose(expected_outputs_file);

        FILE *output_weights_file = fopen(output_weights, 'w');
        if (!output_weights_file) {
            printf("Failed to open %s.\n", output_weights);
            return false;
        }

        FILE *output_bias_file = fopen(output_bias, 'w');
        if (!output_bias_file) {
            printf("Failed to open %s.\n", output_bias);
            return false;
        }




        // Initialise network
        LAYER *layer = (LAYER *) malloc(NUMBER_OF_LAYERS * sizeof(LAYER));
        if (!layer) {
            printf("Failed to allocate memory for layer struct.\n");
            return false;
        }

        // Initialise derivative sum matrices and loop through training data
        mat **weight_matrix_derivative = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
        mat **bias_vector_derivative = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
        mat **weight_matrix_derivative_sum = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
        mat **bias_vector_derivative_sum = (mat **) malloc(NUMBER_OF_LAYERS * sizeof(mat *));
        if (!weight_matrix_derivative_sum || !bias_vector_derivative_sum ||
            !weight_matrix_derivative || !bias_vector_derivative) {
            printf("Failed to initialise derivative sum matrices.\n");
            return false;
        }

        for (size_t i = 0; i < NUMBER_OF_LAYERS; i++) {
            layer[i].neuron_count = NUMBER_OF_NEURONS_IN_LAYER[i];
            if (i != 0) {
                layer[i].weights = mat_init(NUMBER_OF_NEURONS_IN_LAYER[i], NUMBER_OF_NEURONS_IN_LAYER[i - 1]);
                layer[i].bias = mat_init(NUMBER_OF_NEURONS_IN_LAYER[i], 1);
                weight_matrix_derivative_sum[i] = mat_copy(layer[i].weights);
                bias_vector_derivative_sum[i] = mat_copy(layer[i].bias);
                weight_matrix_derivative[i] = mat_copy(layer[i].weights);
                bias_vector_derivative[i] = mat_copy(layer[i].bias);
                initial_weights_selection(layer[i].weights, 0, sqrt(2.0 / (NUMBER_OF_INPUT_NEURONS + NUMBER_OF_EXPECTED_OUTPUTS)));
                initial_weights_selection(layer[i].bias, 0, sqrt(2.0 / (NUMBER_OF_INPUT_NEURONS + NUMBER_OF_EXPECTED_OUTPUTS)));
            }
        }

        // End of data initialisation

        size_t iterations = 0;
        while (iterations < 1000) {
            bool fitted = 0;
            double cost_sum = 0;

            for (size_t i = 0; i < NUMBER_OF_DATA; i++) {

                // select row i from input matrix
                mat *input_layer = mat_select_row(input_matrix, i);
                mat *expected_output_layer = mat_select_row(expected_outputs_matrix, i);
                mat *negative_expected_output_layer = mat_scale(expected_output_layer, -1.0);
                layer[0].neuron = mat_transpose(input_layer);
                mat_free(input_layer);

                mat **z_matrix = (mat**) malloc(NUMBER_OF_LAYERS * sizeof(mat*));


                // forward pass through neural network
                for (size_t j = 1; j < NUMBER_OF_LAYERS; j++) {
                    z_matrix[j] = matZ(layer[j - 1].neuron, &layer[j]);
                    layer[j].neuron = activation_function(mat_copy(z_matrix[j]), activation_function_choice[j]);
                }

                // perform computations to output layer against expected values
                mat *final_layer_neurons = layer[NUMBER_OF_LAYERS - 1].neuron;
                double initial_cost_sum = 0;
                for (size_t j = 0; j < NUMBER_OF_EXPECTED_OUTPUTS; j++) {
                    initial_cost_sum += (mat_select(final_layer_neurons, j, 0) - mat_select(negative_expected_output_layer, 0, j)) * 
                        (mat_select(final_layer_neurons, j, 0) - mat_select(negative_expected_output_layer, 0, j));
                }
                cost_sum += (1 / 2 * NUMBER_OF_EXPECTED_OUTPUTS) * initial_cost_sum;


                // calculate cost function gradients of output layer

                for (size_t j = 0; j < NUMBER_OF_EXPECTED_OUTPUTS; j++) {
                    for (size_t k = 0; k < layer[NUMBER_OF_LAYERS - 2].neuron_count; k++) {
                        double cost_over_layerj = (1 / NUMBER_OF_EXPECTED_OUTPUTS) * (mat_select(final_layer_neurons, j, 0) - mat_select(negative_expected_output_layer, 0, j));
                        double layerj_over_z = mat_select();

                    }

                }



                // backward pass through neural network

                

                // store sum results



            }

            // Compute average cost function

            // Gradient descent

            if (!cost_sum) fitted = 1;
            if (fitted) break;
            iterations++;
        }


        fclose(output_weights_file);
        fclose(output_bias_file);
        return true;
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
