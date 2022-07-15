#include <stdio.h>
#include "linear_model.h"
#include "matrix.h"

int main()
{
	// setup model and its functions
	LinearModel* lm = gmf_model_linear_init();
	lm->activation = &gmf_activation_identity;
	lm->loss = &gmf_loss_squared;
	lm->loss_gradient = &gmf_loss_gradient_squared;

	// setup model parameters
	lm->params->n_iterations = 1000;
	lm->params->learning_rate = 0.00001f;

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
