#include "regularization_gradient.h"
#include "matrix.h"

float gmf_regularization_gradient_L1(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	float weight_sum = 0.0f;
	for (size_t i = 0; i < W->n_rows; ++i)
		weight_sum += mat_at(W_copy, i, 0) / fabsf(mat_at(W_copy, i, 0));
	mat_free(&W_copy);

	return params[0] * weight_sum;
}

float gmf_regularization_gradient_L2(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	float weight_sum = 0.0f;
	for (size_t i = 0; i < W->n_rows; ++i)
		weight_sum += mat_at(W_copy, i, 0); 
	mat_free(&W_copy);

	return params[0] * 2 * weight_sum;
}

float gmf_regularization_gradient_LN(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	float weight_sum = 0.0f;
	for (size_t i = 0; i < W->n_rows; ++i)
		weight_sum += mat_at(W_copy, i, 0); 
	mat_free(&W_copy);

	return params[0] * params[1] * weight_sum;
}
