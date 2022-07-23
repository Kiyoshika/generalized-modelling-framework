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
		if (mat_at(Y, r, 0) > 0.1f && mat_at(Y, r, 0) < 0.6f)
			mat_set(&Y, r, 0, 1.0f);
		else if (mat_at(Y, r, 0) >= 0.6f) // yes, I know >= with floats isn't reliable, but this is a demonstration
			mat_set(&Y, r, 0, 2.0f);
		else
			mat_set(&Y, r, 0, 0.0f);
	}

	// create OVR model with 3 classes
	// passing NULL to class weights will compute them automatically
	// otherwise you can pass an array of floats to control class weights
	LinearModelOVR* ovr_model = gmf_model_linear_ovr_init(3, NULL);

	// set activation/loss/gradient
	gmf_model_linear_ovr_set_activation(&ovr_model, &gmf_activation_sigmoid);
	gmf_model_linear_ovr_set_loss(&ovr_model, &gmf_loss_cross_entropy);
	gmf_model_linear_ovr_set_loss_gradient(&ovr_model, &gmf_loss_gradient_cross_entropy);

	// set n_iterations parameter for all submodels
	// note you can do this manually for each model with ovr_model->models[m]->params->... = ...
	gmf_model_linear_ovr_set_iterations(&ovr_model, 100000);

	gmf_util_add_bias(&X);
	gmf_model_linear_ovr_fit(&ovr_model, X, Y, true);

	printf("\n\nACTUALS:\n");
	mat_print(Y);

	// ovr_model->X already has bias term so we don't have to do it manually
	Matrix* preds = gmf_model_linear_ovr_predict(ovr_model, X); 
	printf("\n\nPREDICTIONS:\n");
	mat_print(preds);

	printf("\n\nCalculated class weights:\n");
	for (size_t i = 0; i < 3; ++i)
		printf("%f\n", ovr_model->class_weights[i]);

	gmf_model_linear_ovr_free(&ovr_model);
	mat_free(&X);
	mat_free(&Y);
	mat_free(&preds);
}
