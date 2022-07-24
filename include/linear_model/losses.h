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

// L(y, yhat) = |y - yhat|
float gmf_loss_absolute(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm);

// L(y, yhat) = max(0, 1 - y * yhat)
// NOTE: hinge loss is originally defined for {-1, 1}
// so we translate 0 to -1 for the purpose of this function.
// For that reason, it is ONLY recommended to use OVR models
// with this loss function as it checks if yhat = 0 whereas
// LinearModel does not convert to hard 0-1 value.
float gmf_loss_hinge(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm);

// L(y, yhat) = {
// 0.5*(y - yhat)^2 for |y - yhat| <= huber_delta,
// huber_delta * (|y - yhat| - 0.5 * huber_delta) otherwise
// }
// NOTE: huber_delta is a param in LinearModel
float gmf_loss_huber(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm);

#endif
