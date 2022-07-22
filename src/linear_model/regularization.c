#include "regularization.h"
#include "matrix.h"

float gmf_regularization_L1(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	for (size_t i = 0; i < W_copy->n_rows; ++i)
		mat_set(&W_copy, i, 0, fabsf(mat_at(W_copy, i, 0)));
	float weight_sum = mat_sum(W_copy);
	mat_free(&W_copy);	

	return params[0] * weight_sum;
}

float gmf_regularization_L2(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	for (size_t i = 0; i < W_copy->n_rows; ++i)
		mat_set(&W_copy, i, 0, powf(mat_at(W_copy, i, 0), 2.0f));
	float weight_sum = mat_sum(W_copy);
	mat_free(&W_copy);	

	return params[0] * weight_sum;
}

float gmf_regularization_LN(const float* params, const Matrix* W)
{
	Matrix* W_copy = mat_copy(W);
	for (size_t i = 0; i < W_copy->n_rows; ++i)
		mat_set(&W_copy, i, 0, powf(mat_at(W_copy, i, 0), params[1]));
	float weight_sum = mat_sum(W_copy);
	mat_free(&W_copy);

	return params[0] * weight_sum;
}
