#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_LINE 1024


typedef struct neuron {
    double *weights;
    double bias;
    int length;
} neuron;

int step(neuron n, double *input);
double dot(neuron n, double *input);


int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("File usage: ./file [input.csv] [output.csv] [model_data.csv] [number of weights]\n");
        return 1;
    }

    FILE *input = fopen(argv[1], "r"); 
    FILE *output = fopen(argv[2], "w");
    FILE *model = fopen(argv[3], "r");
    if (!input || !output || !model) {
        printf("Failed to open file.\n");
        return 2;
    }

    neuron trained;

    trained.length = atoi(argv[4]);
    trained.weights = malloc(sizeof(double) * trained.length);


    char model_line[MAX_LINE];
    fgets(model_line, sizeof(model_line), model);
    char *model_token = strtok(model_line, ",");
    trained.bias = atof(model_token);
    model_token = strtok(NULL, ",");

    int m = 0;
    while (model_token != NULL && m < trained.length) {
        trained.weights[m++] = atof(model_token);
        model_token = strtok(NULL, ",");
    }


    char line[MAX_LINE];
    while (fgets(line, sizeof(line), input)) {
        char *token = strtok(line, ",");
        int g = 0;

        double *x = malloc(sizeof(double) * trained.length);

        while (token != NULL && g < trained.length) {
            x[g++] = atof(token);
            token = strtok(NULL, ",");
        }

        int result = step(trained, x);
        fprintf(output, "%i", result);
        if (g != trained.length - 1) fprintf(output, ",");
        free(x);
    }





    fclose(model);
    fclose(input);
    fclose(output);
    return 0;
}


double dot(neuron n, double *input) {
    double sum = 0;
    for (int i = 0; i < n.length; i++) {
        sum+= n.weights[i] * input[i];
    }

    return sum;
}

int step(neuron n, double *input) {
    double z = dot(n, input) + n.bias;
    
    if (z >= 0) return 1;
    else return 0;
}

