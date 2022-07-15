#ifndef LOSS_GRADIENTS_H
#define LOSS_GRADIENTS_H

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

// L'(y, yhat) = 2(y - yhat)x
void gmf_loss_gradient_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		Matrix** loss_gradient);

#endif
