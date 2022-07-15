#include "loss_gradients.h"
#include "matrix.h"

void gmf_loss_gradient_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		Matrix** loss_gradient)
{
	// compute -2(y - yhat) element-wise and multiply by X
	Matrix* Y_temp = mat_copy(Y);
	mat_subtract_e(&Y_temp, Yhat);
	mat_multiply_s(&Y_temp, -2.0f);
	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, Y_temp, loss_gradient);	
	mat_divide_s(loss_gradient, X->n_rows);
	mat_free(&Y_temp);
	mat_free(&X_t);
}
