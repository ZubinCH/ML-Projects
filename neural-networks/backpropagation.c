#include <stdlib.h>
#include <stdio.h>

#define LAYERS_NUMBER 5


typedef struct NEURON {
    double *weights;
    double bias;
} NEURON;

typedef struct LAYER {
    NEURON *neuron;
    const unsigned int NUMBER_OF_NEURONS;
} LAYER;

typedef struct network {
    LAYER *layers;
    const unsigned int NUMBER_OF_LAYERS;
} network;