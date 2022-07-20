#include "loss_gradients.h"
#include "matrix.h"

void gmf_loss_gradient_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const size_t* class_pair,
		const float* class_weights,
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

void gmf_loss_gradient_cross_entropy(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const size_t* class_pair,
		const float* class_weights,
		Matrix** loss_gradient)
{
	// compute (yhat - y) element-wise and multiply by X
	Matrix* Yhat_temp = mat_copy(Yhat);
	mat_subtract_e(&Yhat_temp, Y);

	// apply class weights
	if (class_weights)
	{
		for (size_t r = 0; r < Y->n_rows; ++r)
			mat_set(&Yhat_temp, r, 0, mat_at(Yhat_temp, r, 0) * class_weights[class_pair[(size_t)mat_at(Y, r, 0)]]);
	}

	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, Yhat_temp, loss_gradient);
	mat_divide_s(loss_gradient, X->n_rows);
	mat_free(&Yhat_temp);
	mat_free(&X_t);
}
