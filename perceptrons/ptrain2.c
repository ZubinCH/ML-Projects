#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LOOPS 1000
#define MAX_LINE 1024
#define LEARNING_RATE 0.1


typedef struct vec {
    double *col;
    int length;
} vec;

typedef struct data {
    vec *x;
    int y;
} data;

typedef struct neuron {
    vec *weights;
    double bias;
} neuron;

typedef struct layer {
    neuron *n;
    int length;
} layer;


// math functions' prototypes
double dot(vec *vec1, vec *vec2);
double relu(double z);
int step(double z);
int output(neuron *n, vec *x);
void train(neuron *n, data *d, double yhat);

// memory and print functions' prototypes
void print_vec(vec *vec1);

void neuron_init(neuron *n, int length);
void free_neuron(neuron *n);

void vec_init(vec *vec1, int length);
void free_vec(vec *v);

data *data_fill(data *d, char *l, int length);
void free_data(data *d);

void save_neuron(FILE *file, neuron *n);






int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Usage: ./file [input.csv] [output.csv] [number of inputs]\n");
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    FILE *trained = fopen(argv[2], "w");
    if (input == NULL || trained == NULL) {
        printf("Could not open file.\n");
        return 1;
    }

    int inputs_number = atoi(argv[3]);

    int iterations = 0;

    neuron *neu = malloc(sizeof(neuron));
    if (!neu) return 1;
    neuron_init(neu, inputs_number);

    for (int i = 0; i < MAX_LOOPS; i++) {
        int fitted = 1;
        char line[MAX_LINE];
        rewind(input);

        while(fgets(line, sizeof(line), input)) {
            line[strcspn(line, "\n")] = '\0';

            data *d = malloc(sizeof(data));
            d = data_fill(d, line, inputs_number);

            if (d->y != output(neu, d->x)) {
                train(neu, d, output(neu, d->x));
                fitted = 0;
            }

            free_data(d);
        }

        iterations++;
        if (fitted) break;
    }

    printf("Iterations: %i\n", iterations);
    printf("Weights: ");
    print_vec(neu->weights);
    printf("Bias: %0.3f\n", neu->bias);
    save_neuron(trained, neu);



    free_neuron(neu);
    return 0;
}










double dot(vec *vec1, vec *vec2) {
    if (vec1->length != vec2->length) {
        printf("Vectors are not the same length.\n");
        return 0;
    }

    double sum = 0;
    for (int i = 0, n = vec1->length; i < n; i++) {
        sum+= vec1->col[i] * vec2->col[i];
    }

    return sum;
}


double relu(double z) {
    if (z <= 0) return 0;
    else return z;   
}

int step(double z) {
    if (z >= 0) return 1;
    else return 0;
}

int output(neuron *n, vec *x) {
    double z = dot(n->weights, x) + n->bias;
    return step(z);
}


void train(neuron *n, data *d, double yhat) {
    if (n->weights->length != d->x->length) {
        printf("Vectors do not have same length");
        return;
    }

    for (int i = 0, m = n->weights->length; i < m; i++) {
        n->weights->col[i] += LEARNING_RATE * (d->y - yhat) * d->x->col[i];
    }
    n->bias += LEARNING_RATE * (d->y - yhat);
}

void print_vec(vec *vec1) {
    printf("[");
    for (int i = 0, m = vec1->length; i < m; i++) {
        printf("%0.3f", vec1->col[i]);
        if (i != m - 1) printf(", ");
    }
    printf("]\n");
}

void neuron_init(neuron *n, int length) {
    n->weights = malloc(sizeof(vec));
    if (!n->weights) return;

    vec_init(n->weights, length);
    n->bias = 0;
}


void vec_init(vec *vec1, int length) {
    vec1->col = calloc(length, sizeof(double));
    if (!vec1->col) {
        printf("Initialisation of vector failed.\n");
        return;
    }
    vec1->length = length;
}


void free_vec(vec *v) {
    free(v->col);
    return;
}

void free_neuron(neuron *n) {
    free_vec(n->weights);
    free(n->weights);
    free(n);
    return;
}


data *data_fill(data *d, char *l, int length) {

    char *token = strtok(l, ",");
    if (!token) {
        printf("Tokenisation failed.\n");
        return d;
    }

    d->y = atoi(token);
    token = strtok(NULL, ",");

    int m = 0;
    d->x = malloc(sizeof(vec));
    vec_init(d->x, length);

    while (token != NULL && m < length) {
        d->x->col[m++] = atof(token);
        token = strtok(NULL, ",");
    }


    return d;
}

void free_data(data *d) {
    free_vec(d->x);
    free(d->x);
    free(d);
}

void save_neuron(FILE *file, neuron *n) {
    fprintf(file, "%0.3f,", n->bias);
    for (int i = 0, m = n->weights->length; i < m; i++) {
        fprintf(file, "%0.3f", n->weights->col[i]);
        if (i != m - 1) fprintf(file, ",");
    }
}