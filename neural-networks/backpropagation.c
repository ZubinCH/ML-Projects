
/*
 * neural-network.c
 * Author: Zubin Chowdhury (GitHub: @ZubinCH)
 * Date: 19.11.2026
 * Purpose: A scalable neural network program built in C. 
 * It can handle any any number of inputs, number of layers (within reason), neurons per layer, and provide the desired number of outputs.
 * Only the training function has been built so far.
 * 
 * 
 * 
 * 
 * 
 * Active to do list:
    * Complete the accuracy check function for supervised learning after data has been fitted
        * Ensure check files have data the model hasn't been trained on
    * Build input and expected output files for datasets
    * Implement SQL based file handling rather than using CSV files
    * Implement regularisation of weights and biases into the algorithm to prevent overfitting
    * Implement apply function where a trained neural network can be used for classifying and predicting data
    * Implement time checks for different components of the program to see how long each section of the program takes and what can be optimised
    * Implement int main void in a separate file
    * Build CI/CD tests to verify all components of the program work as intended
        * mkdir file_handling
 *          
 *      
 *      
 * 
 *
*/

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "backpropagation.h"
#include "libraries/linear-algebra/matrices.h"
#include "libraries/random-numbers/random_numbers.h"


double activation_function (double z, char function_choice) {
    switch (function_choice) {
        case 's':
            if (z < 0) return 0;
            return 1;
        case 'o':
            return (1.0 / (1.0 + exp(-z)));
        case 'r':
            if (z < 0) return 0;
            return z;
        default:
            printf("Invalid function choice.\n");
            return NAN;
    }
}

double activation_function_derivative(double z, char function_choice) {
    switch (function_choice) {
        case 's':
            return 0;
        case 'o':
            return (exp(-z) / ((1.0 + exp(-z)) * (1.0 + exp(-z))));
        case 'r':
            if (z < 0) return 0;
            return 1;
        default:
            printf("Invalid function choice.\n");
            return NAN;
    }
}

void mat_apply_activation_function(mat *z, char function_choice) {
    size_t total_components = z->rows * z->cols;
    for (size_t i = 0; i < total_components; i++) {
        z->arr[i] = activation_function(z->arr[i], function_choice);
    }

}

void mat_apply_activation_function_derivative(mat *z, char function_choice) {
    size_t total_components = z->rows * z->cols;
    for (size_t i = 0; i < total_components; i++) {
        z->arr[i] = activation_function_derivative(z->arr[i], function_choice);
    }

}


void network_free(LAYER *layer, size_t count) {
    for (size_t i = 1; i < count; i++) {
        mat_free(layer[i].weights);
        mat_free(layer[i].bias);
        mat_free(layer[i].neuron);
    }
    
    free(layer);
}


mat *matZ(mat *inputs, LAYER *layer) {
    if (inputs->cols != 1) {
        printf("Invalid vector.\n");
        return NULL;
    }

    mat *z = mat_mult(layer->weights, inputs);
    size_t total_elements = z->cols * z->rows;

    for (size_t i = 0; i < total_elements; i++) {
        z->arr[i] += layer->bias->arr[i];
    }

    return z;
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
        line[strcspn(line, "\n")] = '\0';
        char *token = strtok(line, ",");
        if (!token) {
            printf("Failed to tokenise.\n");
            return;
        }

        size_t n = 0;
        while (token && n < n_max) {
            double value = atof(token);
            mat_input(value, m, n, mt);
            token = strtok(NULL, ",");
            n++;
        }
        m++;
    }
    return;
}

void load_mat_array_with_file(size_t NUMBER_OF_MATRICES, mat **matrices, FILE *file) {
    char line[MAX_LINE];
    size_t i = 0;

    while (fgets(line, sizeof(line), file) && i < NUMBER_OF_MATRICES) {
        size_t m_max = matrices[i]->rows;
        size_t n_max = matrices[i]->cols;

        line[strcspn(line, "\n")] = '\0';
        char *token = strtok(line, ",");
        if (!token) {
            printf("Failed to tokenise.\n");
            return;
        }

        for (size_t m = 0; m < m_max; m++) {
            size_t n = 0;
            while (n < n_max && token) {
                double value = atof(token);
                mat_input(value, m, n, matrices[i]);
                token = strtok(NULL, ",");
                n++;
            }


        }
        i++;
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
        // Initialise all variables that must be freed by cleanup
        mat *input_matrix = NULL;
        mat *expected_outputs_matrix = NULL;
        LAYER *layer = NULL;
        mat **weight_matrix_derivative_sum = NULL;
        mat **bias_vector_derivative_sum = NULL;
        mat *input_layer = NULL;
        mat *expected_output_layer = NULL;
        mat **matrix_z = NULL;
        mat *input_layer = NULL;

        // Check for whether the function successfully performed all necessary operations
        bool status = false;


        // set new seed for initialising matrices' normally distributed random number components
        srand((unsigned int) time(NULL));

        // Initialise constants
        const size_t NUMBER_OF_EXPECTED_OUTPUTS = NUMBER_OF_NEURONS_IN_LAYER[NUMBER_OF_LAYERS - 1];
        const size_t NUMBER_OF_INPUT_NEURONS = NUMBER_OF_NEURONS_IN_LAYER[0];

        // Open input data and expected output files

        // Load inputs, tokenise and store into matrix
        FILE *input_file = fopen(input_data, "r");
        if (!input_file) {
            printf("Failed to open %s.\n", input_data);
            return false;
        }
        input_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_NEURONS_IN_LAYER[0]);
        if (!input_matrix) {
            fclose(input_file);
            return false;
        }
        load_matrix_with_file(input_matrix, input_file);
        fclose(input_file);

        // Load expected outputs, tokenise and store into matrix
        FILE *expected_outputs_file = fopen(expected_outputs, "r");
        if (!expected_outputs_file) {
            printf("Failed to open %s.\n", expected_outputs);
            mat_free(input_matrix);
            return false;
        }
        expected_outputs_matrix = mat_init(NUMBER_OF_DATA, NUMBER_OF_EXPECTED_OUTPUTS);
        if (!expected_outputs_matrix) {
            fclose(expected_outputs_file);
            goto cleanup;
        }
        load_matrix_with_file(expected_outputs_matrix, expected_outputs_file);
        fclose(expected_outputs_file);


        // Initialise network
        layer = (LAYER *) calloc(NUMBER_OF_LAYERS, sizeof(LAYER));
        if (!layer) {
            printf("Failed to allocate memory for layer struct.\n");
            goto cleanup;
        }

        // Initialise derivative sum matrices
        weight_matrix_derivative_sum = (mat **) calloc(NUMBER_OF_LAYERS, sizeof(mat *));
        bias_vector_derivative_sum = (mat **) calloc(NUMBER_OF_LAYERS, sizeof(mat *));

        if (!weight_matrix_derivative_sum || !bias_vector_derivative_sum) {
            printf("Failed to initialise derivative sum matrices.\n");
            goto cleanup;
        }

        for (size_t i = 1; i < NUMBER_OF_LAYERS; i++) {
            layer[i].weights = mat_init(NUMBER_OF_NEURONS_IN_LAYER[i], NUMBER_OF_NEURONS_IN_LAYER[i - 1]);
            layer[i].bias = mat_init(NUMBER_OF_NEURONS_IN_LAYER[i], 1);

            if (!layer[i].weights || !layer[i].bias) {
                printf("Failed to initialise layer weights and biases.\n");
                goto cleanup;
            }

            weight_matrix_derivative_sum[i] = mat_copy(layer[i].weights);
            bias_vector_derivative_sum[i] = mat_copy(layer[i].bias);
            if (!weight_matrix_derivative_sum[i] || !bias_vector_derivative_sum[i]) {
                printf("Failed to copy matrix.\n");
                goto cleanup;
            }

            initial_weights_selection(layer[i].weights, 0, sqrt(2.0 / (double) (NUMBER_OF_INPUT_NEURONS + NUMBER_OF_EXPECTED_OUTPUTS)));
            initial_weights_selection(layer[i].bias, 0, sqrt(2.0 / (double) (NUMBER_OF_INPUT_NEURONS + NUMBER_OF_EXPECTED_OUTPUTS)));
        }

        // End of data initialisation

        size_t iterations = 0;
        while (iterations < 1000) {
            bool fitted = 0;
            double cost_sum = 0;

            // Ensure all derivative matrices and vectors are zero before looping through all training data 
            for (size_t L = 1; L < NUMBER_OF_LAYERS; L++) {
                for (size_t j = 0; j < NUMBER_OF_NEURONS_IN_LAYER[L]; j++) {
                    for (size_t k = 0; k < NUMBER_OF_NEURONS_IN_LAYER[L - 1]; k++) {
                        mat_input(0, j, k, weight_matrix_derivative_sum[L]);
                    }
                    mat_input(0, j, 0, bias_vector_derivative_sum[L]);
                }

            }

            // Loop through all training data
            for (size_t i = 0; i < NUMBER_OF_DATA; i++) {

                // select row i from input matrix and expected output matrix
                input_layer = mat_select_row(input_matrix, i);
                expected_output_layer = mat_select_row(expected_outputs_matrix, i);
                if (!input_layer || !expected_output_layer) goto cleanup;
                layer[0].neuron = mat_transpose(input_layer);
                if (!layer[0].neuron) goto cleanup;
                mat_free(input_layer);
                if (!layer[0].neuron) goto cleanup;


                matrix_z = (mat**) calloc(NUMBER_OF_LAYERS, sizeof(mat*));

                // forward pass through neural network
                for (size_t j = 1; j < NUMBER_OF_LAYERS; j++) {
                    matrix_z[j] = matZ(layer[j - 1].neuron, &layer[j]);
                    layer[j].neuron = mat_copy(matrix_z[j]);
                    if (!layer[j].neuron) goto cleanup;
                    mat_apply_activation_function(layer[j].neuron, activation_function_choice[j]);
                    mat_apply_activation_function_derivative(matrix_z[j], activation_function_choice[j]);
                }

                // perform computations to output layer against expected values
                mat *final_layer_neurons = layer[NUMBER_OF_LAYERS - 1].neuron;
                double initial_cost_sum = 0;
                for (size_t j = 0; j < NUMBER_OF_EXPECTED_OUTPUTS; j++) {
                    initial_cost_sum += (mat_select(final_layer_neurons, j, 0) - mat_select(expected_output_layer, 0, j)) * 
                        (mat_select(final_layer_neurons, j, 0) - mat_select(expected_output_layer, 0, j));
                }
                cost_sum += (1.0 / (2.0 * (double) NUMBER_OF_EXPECTED_OUTPUTS) ) * initial_cost_sum;


                // calculate cost function gradients of output layer

                double cost_over_final_layer[NUMBER_OF_EXPECTED_OUTPUTS];

                for (size_t j = 0; j < NUMBER_OF_EXPECTED_OUTPUTS; j++) {
                    double layerj_over_z = 0;
                    double z_over_weights = 0;

                    for (size_t k = 0; k < NUMBER_OF_NEURONS_IN_LAYER[NUMBER_OF_LAYERS - 2]; k++) {
                        cost_over_final_layer[j] = (1.0 / (double) NUMBER_OF_EXPECTED_OUTPUTS) * (mat_select(final_layer_neurons, j, 0) - mat_select(expected_output_layer, 0, j));
                        layerj_over_z = mat_select(matrix_z[NUMBER_OF_LAYERS - 1], j, 0);
                        z_over_weights = mat_select(final_layer_neurons, k, 0);

                        double product_of_derivatives = cost_over_final_layer[j] * layerj_over_z * z_over_weights;

                        mat_input(mat_select(weight_matrix_derivative_sum[NUMBER_OF_LAYERS - 1], j, k) + product_of_derivatives, j, k, weight_matrix_derivative_sum[NUMBER_OF_LAYERS - 1]);
                    }
                    mat_input(mat_select(bias_vector_derivative_sum[NUMBER_OF_LAYERS - 1], j, 0) + cost_over_final_layer[j] * layerj_over_z, j, 0, bias_vector_derivative_sum[NUMBER_OF_LAYERS - 1]);
                }


                // backward pass through neural network

                double previous_layer_neuron_sum[NUMBER_OF_LAYERS - 1];

                for (int L = NUMBER_OF_LAYERS - 2; L > 0; L--) {
                    for (size_t k = 0; k < NUMBER_OF_NEURONS_IN_LAYER[L]; k++) {

                        previous_layer_neuron_sum[L - 1] = 0;
                        if (L == NUMBER_OF_LAYERS - 2) {
                            for (size_t j = 0; j < NUMBER_OF_EXPECTED_OUTPUTS; j++) {
                                previous_layer_neuron_sum[L - 1] += cost_over_final_layer[j] 
                                * mat_select(matrix_z[L + 1], j, 0) * mat_select(layer[L + 1].weights, j, k);
                            }
                        }

                        else {
                            for (size_t j = 0; j < NUMBER_OF_NEURONS_IN_LAYER[L + 1]; j++) {
                                previous_layer_neuron_sum[L - 1] += 
                                mat_select(matrix_z[L + 1], j, 0) * mat_select(layer[L + 1].weights, j, k);
                            }
                            previous_layer_neuron_sum[L - 1] *= previous_layer_neuron_sum[L];
                        }

                        for (size_t j = 0; j < NUMBER_OF_NEURONS_IN_LAYER[L + 1]; j++) {
                            mat_input(mat_select(weight_matrix_derivative_sum[L], j, k) 
                            + previous_layer_neuron_sum[L - 1] * mat_select(matrix_z[L], k, 0) 
                            * mat_select(layer[L].neuron, k, 0),
                            j, k, weight_matrix_derivative_sum[L]);
                        }

                        mat_input(mat_select(bias_vector_derivative_sum[L], k, 0) 
                            + previous_layer_neuron_sum[L - 1] * mat_select(matrix_z[L], k, 0),
                            k, 0, bias_vector_derivative_sum[L]);
                    }
                }

                mat_free(layer[0].neuron);
            }

            // Compute average cost function

            for (size_t L = 1; L < NUMBER_OF_LAYERS; L++) {
                size_t total_elements = NUMBER_OF_NEURONS_IN_LAYER[L] * NUMBER_OF_NEURONS_IN_LAYER[L - 1];
                for (size_t i = 0; i < total_elements; i++) {
                    weight_matrix_derivative_sum[L]->arr[i] *= 1.0 / (double) (NUMBER_OF_DATA);
                    layer[L].weights->arr[i] -= LEARNING_RATE * weight_matrix_derivative_sum[L]->arr[i];
                }

                for (size_t j = 0; j < NUMBER_OF_NEURONS_IN_LAYER[L]; j++) {
                    bias_vector_derivative_sum[L]->arr[j] *= 1.0 / (double) (NUMBER_OF_DATA);
                    layer[L].bias->arr[j] -= LEARNING_RATE * bias_vector_derivative_sum[L]->arr[j];
                }
            }

            cost_sum = cost_sum / (double) NUMBER_OF_DATA;
            if (fabs(cost_sum) < 0.1) fitted = 1;
            if (fitted) break;
            iterations++;
        }


        // Output weights and biases into output files
        FILE *output_weights_file = fopen(output_weights, "w");
        FILE *output_bias_file = fopen(output_bias, "w");
        if (!output_weights_file) {
            printf("Failed to open output weights file.\n");
            goto cleanup;
        }
        if (!output_bias_file) {
            printf("Failed to open output bias file.\n");
            fclose(output_weights_file);
            goto cleanup;
        }



        // Store each weight matrix as a single line - later initialisation with other functions will calculate the matrix sizes using neurons per layer
        for (size_t L = 1; L < NUMBER_OF_LAYERS; L++) {
            size_t total_elements = layer[L].weights->rows * layer[L].weights->cols;
            size_t total_vector_elements = layer[L].bias->rows * layer[L].bias->cols;
            for (size_t j = 0; j < total_elements; j++) {
                fprintf(output_weights_file, "%0.10f", layer[L].weights->arr[j]);
                if (j != total_elements - 1) fprintf(output_weights_file, ",");
            }

            for (size_t k = 0; k < total_vector_elements; k++) {
                fprintf(output_bias_file, "%0.10f", layer[L].bias->arr[k]);
                if (k != total_vector_elements - 1) fprintf(output_bias_file, ",");
            }

            fprintf(output_bias_file, "\n");
            fprintf(output_weights_file, "\n");
        }

        fclose(output_weights_file);
        fclose(output_bias_file);
        status = true;

        cleanup:
            mat_free(input_layer);
            mat_array_free(matrix_z, NUMBER_OF_LAYERS);
            mat_free(expected_output_layer);
            mat_free(input_layer);
            mat_array_free(bias_vector_derivative_sum, NUMBER_OF_LAYERS);
            mat_array_free(weight_matrix_derivative_sum, NUMBER_OF_LAYERS);
            network_free(layer, NUMBER_OF_LAYERS);
            mat_free(expected_outputs_matrix);
            mat_free(input_matrix);
            return status;
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
