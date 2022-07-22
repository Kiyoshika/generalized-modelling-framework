#ifndef LOSSES_H
#define LOSSES_H

#include <math.h>

// forward declaration
typedef struct Matrix Matrix;
typedef struct LinearModel LinearModel;

// L(y, yhat) = (y - yhat)^2
float gmf_loss_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm);

// L(y, yhat) = -ylog(yhat) - (1 - y)log(1 - yhat)
float gmf_loss_cross_entropy(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm);

#endif
