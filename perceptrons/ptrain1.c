#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LOOPS 10000
#define MAX_LINE 1024

// Function prototypes
double dot(double *vec1, double *vec2, int length);
int step(double z);

void print_arr(double *arr, int length);


int main(int argc, char *argv[]) {

    // Check for valid arguments

    if (argc != 3) {
        printf("File use: ./file [inputs.csv] [number of inputs]\n");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Could not open file.\n");
        return 1;
    }


    // Initialise variables
    int inputs_number = atoi(argv[2]);
    double *weights = calloc(inputs_number, sizeof(double));
    if (!weights) {
        printf("Memory allocation failed\n");
        return 1;
    }
    double bias = 0;
    float learning_rate = 0.01;


    // Main loop
    for (int i = 0; i < MAX_LOOPS; i++) {
        int fitted = 1;
        char line[MAX_LINE];
        rewind(file);
        while (fgets(line,sizeof(line), file)) {

            line[strcspn(line, "\n")] = '\0';
            
            // Tokenisation of lines via commas
            char *token = strtok(line, ",");
            if (!token) continue;
            int y = atoi(token);
            token = strtok(NULL, ",");

            double x[inputs_number];
            int n = 0;

            while (token != NULL && n < inputs_number) {
                x[n++] = atof(token);
                token = strtok(NULL, ",");
            }

            // Perform f(x dot w + bias)
            double z = dot(weights, x, inputs_number) + bias;
            int yhat = step(z);

            // Change weights and bias if algorithm's output is not expected output
            if (yhat != y) {
               for (int j = 0; j < inputs_number; j++) {
                    weights[j] += learning_rate * (y - yhat) * x[j];
                }
                
                bias += learning_rate * (y - yhat);
                fitted = 0;
            }

        }

        if (fitted) break;
    }

    // Print data obtained from training
    printf("Weights: ");
    print_arr(weights, inputs_number);
    printf("Bias %0.3f\n", bias);


    free(weights);
    fclose(file);
    return 0;
}


double dot(double *vec1, double *vec2, int length) {
    if (length < 1) {
        printf("Vector too small for dot product.\n");
        return 0;
    }

    double sum = 0;

    for (int i = 0; i < length; i++) {
        sum+= vec1[i] * vec2[i];
    }

    return sum;
}


void print_arr(double *arr, int length) {
    printf("[");

    for (int i = 0; i < length - 1; i++) {
        printf("%0.3f, ", arr[i]);
    }
    printf("%0.3f]\n", arr[length - 1]);

}

int step(double z) {
    if (z >= 0) return 1;
    else return 0;
}

