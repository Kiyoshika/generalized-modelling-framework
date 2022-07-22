#ifndef REGULARIZATION_GRADIENT_H
#define REGULARIZATION_GRADIENT_H

#include <math.h>
#include <stddef.h>

// forward declaration
typedef struct Matrix Matrix;

// params[0] * sum(w_i / |w_i|)
float gmf_regularization_gradient_L1(const float* params, const Matrix* W);

// params[0] * 2 * sum(w_i)
float gmf_regularization_gradient_L2(const float* params, const Matrix* W);

// params[0] * params[1] * sum(w_i)
float gmf_regularization_gradient_LN(const float* params, const Matrix* W);

#endif
