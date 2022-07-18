#include "linear_model_ovr.h"
#include "matrix.h"
#include <stdio.h>

int main()
{
	Matrix* X = NULL;
	mat_init(&X, 20, 5);
	mat_random(&X, -5.0f, 5.0f);

	Matrix* Y = NULL;
	mat_init(&Y, 20, 1);
	mat_random(&Y, 0.0f, 1.0f);

	// generate fake multiclasses [0, 1, 2]
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		if (mat_at(Y, r, 0) > 0.3f && mat_at(Y, r, 0) < 0.6f)
			mat_set(&Y, r, 0, 1.0f);
		else if (mat_at(Y, r, 0) >= 0.6f) // yes, I know >= with floats isn't reliable, but this is a demonstration
			mat_set(&Y, r, 0, 2.0f);
		else
			mat_set(&Y, r, 0, 0.0f);
	}

	// create OVR model with 3 classes
	LinearModelOVR* ovr_model = gmf_model_linear_ovr_init(3);

	// manually set activation/loss/gradient (need to finish this in API)
	for (size_t m = 0; m < ovr_model->n_models; ++m)
	{
		ovr_model->models[m]->activation = &gmf_activation_sigmoid;
		ovr_model->models[m]->loss = &gmf_loss_cross_entropy;
		ovr_model->models[m]->loss_gradient = &gmf_loss_gradient_cross_entropy;
	}

	gmf_model_linear_ovr_fit(&ovr_model, X, Y);

	gmf_model_linear_ovr_free(&ovr_model);
}
