#ifndef GMF_UTIL_H
#define GMF_UTIL_H

// forward declaration
typedef struct Matrix Matrix;

// add a bias term - modifies X inplace and potentially
// changes the underlying pointer address
void gmf_util_add_bias(Matrix** X);

#endif
