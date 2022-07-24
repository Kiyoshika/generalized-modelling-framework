#include "loss_gradients.h"
#include "linear_model.h"
#include "matrix.h"

static void __add_regularization(const LinearModel* lm, Matrix** Y)
{
	if (lm->regularization_gradient)
	{
		float regularization = lm->regularization_gradient(lm->params->regularization_params, lm->W);
		mat_add_s(Y, regularization);
	}
}

void gmf_loss_gradient_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient)
{
	// compute -2(y - yhat) element-wise and multiply by X
	Matrix* Y_temp = mat_copy(Y);
	mat_subtract_e(&Y_temp, Yhat);
	mat_multiply_s(&Y_temp, -2.0f);
	__add_regularization(lm, &Y_temp);	

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
		const LinearModel* lm,
		Matrix** loss_gradient)
{
	// compute (yhat - y) element-wise and multiply by X
	Matrix* Yhat_temp = mat_copy(Yhat);
	mat_subtract_e(&Yhat_temp, Y);

	// apply class weights
	if (lm->params->class_weights)
	{
		for (size_t r = 0; r < Y->n_rows; ++r)
			mat_set(&Yhat_temp, r, 0, mat_at(Yhat_temp, r, 0) * lm->params->class_weights[lm->params->class_pair[(size_t)mat_at(Y, r, 0)]]);
	}

	__add_regularization(lm, &Yhat_temp);

	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, Yhat_temp, loss_gradient);
	mat_divide_s(loss_gradient, X->n_rows);
	mat_free(&Yhat_temp);
	mat_free(&X_t);
}

void gmf_loss_gradient_absolute(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient)
{
	Matrix* result = NULL;
	mat_init(&result, Y->n_rows, 1);

	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);

		if (fabsf(y - yhat) < 0.0001f)
			mat_set(&result, r, 0, 0.0f);
		else
			mat_set(&result, r, 0, (y - yhat)/fabsf(y - yhat));
	}
	
	mat_multiply_s(&result, -1.0f); // carry negative from chain rule

	__add_regularization(lm, &result);

	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, result, loss_gradient);
	mat_divide_s(loss_gradient, X->n_rows);

	mat_free(&result);
	mat_free(&X_t);
}

void gmf_loss_gradient_hinge(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient)
{
	Matrix* result = NULL;
	mat_init(&result, Y->n_rows, 1);

	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);

		// yeah, <= prob not great for floats, but good enough
		if (y * yhat <= 0.0f)
			mat_set(&result, r, 0, 0.5f - y * yhat);
		else if (y * yhat > 0.0f && y * yhat <= 1.0f)
			mat_set(&result, r, 0, 0.5f * (1.0f - y * yhat));
		else
			mat_set(&result, r, 0, 0.0f);
	}

	__add_regularization(lm, &result);

	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, result, loss_gradient);
	mat_divide_s(loss_gradient, X->n_rows);

	mat_free(&result);
	mat_free(&X_t);
}

void gmf_loss_gradient_huber(
		const Matrix* Y,
		const Matrix* Yhat,
		const Matrix* X,
		const LinearModel* lm,
		Matrix** loss_gradient)
{
	Matrix* result = NULL;
	mat_init(&result, Y->n_rows, 1);
	
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);

		if (fabsf(y - yhat) <= lm->params->huber_delta)
			mat_set(&result, r, 0, y - yhat);
		else
			mat_set(&result, r, 0, lm->params->huber_delta * ((y - yhat) / fabsf(y - yhat)));
	}

	mat_multiply_s(&result, -1.0f); // carry negative from chain rule

	__add_regularization(lm, &result);

	Matrix* X_t = mat_transpose(X);
	mat_multiply_inplace(X_t, result, loss_gradient);
	mat_divide_s(loss_gradient, X->n_rows);

	mat_free(&result);
	mat_free(&X_t);
}
