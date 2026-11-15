#ifndef RANDOM_NUMBERS_H
#define RANDOM_NUMBERS_H

#include <stdbool.h>
#include <stddef.h>

#define PI 3.14159265358979

// Function prototypes
double normalised_random_number(bool has_sign);
double normal_pdf(double x, double mean, double stdv);
double normally_distributed_random_number(double mean, double stdv);
double average(double *arr, size_t size);
double sample_stdv(double *arr, double mean, size_t size);



#endif