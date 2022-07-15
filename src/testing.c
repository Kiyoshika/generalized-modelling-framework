#include <stdio.h>
#include "linear_model.h"
#include "matrix.h"

int main()
{
	LinearModelParams params;
	params.n_iterations = 10000;

	LinearModel* lm = gmf_model_linear_init(params);
	lm->activation = &gmf_activation_identity;
	lm->loss = &gmf_loss_squared;
	lm->loss_gradient = &gmf_loss_gradient_squared;

	Matrix* X = NULL;
	mat_init(&X, 10000, 50);
	mat_random(&X, 0.0f, 10.0f);

	Matrix* Y = NULL;
	mat_init(&Y, 10000, 1);
	mat_random(&Y, 2.0f, 15.0f);

	gmf_model_linear_fit(&lm, X, Y);

	gmf_model_linear_free(&lm);
	mat_free(&X);
	mat_free(&Y);

	return 0;
}
