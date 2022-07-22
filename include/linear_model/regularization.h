#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include <math.h>
#include <stddef.h>

// forward declaration
typedef struct Matrix Matrix;

// params[0] * sum(|w_i|)
float gmf_regularization_L1(const float* params, const Matrix* W);

// params[0] * sum((w_i)^2)
float gmf_regularization_L2(const float* params, const Matrix* W);

// params[0] * sum((w_i)^params[1])
float gmf_regularization_LN(const float* params, const Matrix* W);

#endif
