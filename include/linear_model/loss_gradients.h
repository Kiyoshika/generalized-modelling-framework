#ifndef LOSS_GRADIENTS_H
#define LOSS_GRADIENTS_H

#include <stdint.h>
#include <stddef.h>

/*
 * NOTE: we are given Y and Yhat (both of shape (r, 1))
 * and our input data X. Typically the loss is finally
 * multiplied with X giving a (c, 1) column vector that
 * we can use to update the weights.
 *
 * It is assumed that loss_gradient matrix is pre-allocated
 * prior to calling any of these functions.
 */

// formward declaration
typedef struct Matrix Matrix;
typedef struct LinearModel LinearModel;

// L'(y, yhat) = 2(y - yhat)x
void gmf_loss_gradient_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient);

// L'(y, yhat) = (yhat - y)x
void gmf_loss_gradient_cross_entropy(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient);

// L'(y, yhat) = (y - yhat)/|y - yhat|
// or L'(y, yhat) = sgn(y - yhat)
void gmf_loss_gradient_absolute(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient);

// L'(y, yhat) = {
// 0.5 - y * yhat if y * yhat <= 0
// 0.5 * (1 - y * yhat) if 0 < y * yhat < 1
// 0 if 1 <= y * yhat
// }
// NOTE: this is the "smoothed" version of the hinge gradient
void gmf_loss_gradient_hinge(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient);

// L'(y, yhat) = {
// (y - yhat) if |y - yhat| <= huber_delta
// huber_delta * sgn(y - yhat) for |y - yhat| > huber_delta
// }
void gmf_loss_gradient_huber(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient);

#endif
