#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stddef.h>

/*
 * NOTE:
 * For all activations, we can assume we are given the
 * linear combination of X and W. So all parameters will
 * be XW that have shape (r, 1)
 */

// forward declaration
typedef struct Matrix Matrix;

// f(x) = x
void gmf_activation_identity(Matrix** XW);

// f(x) = 1 / (1 + e^(-x))
void gmf_activation_sigmoid(Matrix** XW);

#endif
