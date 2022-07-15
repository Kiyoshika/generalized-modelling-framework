#ifndef LOSSES_H
#define LOSSES_H

// forward declaration
typedef struct Matrix Matrix;

// L(y, yhat) = (y - yhat)^2
float gmf_loss_squared(
		const Matrix* Y,
		const Matrix* Yhat);

#endif
