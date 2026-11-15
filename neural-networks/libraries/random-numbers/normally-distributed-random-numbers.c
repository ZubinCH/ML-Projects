#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "random_numbers.h"


double normalised_random_number(bool has_sign) {
    double value = (double)rand() / RAND_MAX;
    int sign = (has_sign) ? pow(-1, rand() % 2) : 1;

    return value * (double) sign;
}

double normal_pdf(double x, double mean, double stdv) {
    return (1.0 / (stdv * sqrt(2.0 * PI)) * exp(-0.5 * ((x - mean) / stdv) * ((x - mean) / stdv)));
}

double normally_distributed_random_number(double mean, double stdv) {
    double value = mean + 5 * stdv * normalised_random_number(true);
    double random_probability = normalised_random_number(false);

    while (random_probability > normal_pdf(value, mean, stdv)) {
        random_probability = normalised_random_number(false);
        value = mean + 5 * stdv * normalised_random_number(true);        
    }

    return value;
}

double average(double *arr, size_t size) {
    if (size == 1) return *arr;

    double sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += arr[i];
    }

    return sum / (double) size;
}

double sample_stdv(double *arr, double mean, size_t size) {
    if (size == 1) return 0.0;

    double sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += (arr[i] - mean) * (arr[i] - mean);

    }
    if (sum < 0) sum *= -1.0;

    return sqrt(sum / (size - 1));
}