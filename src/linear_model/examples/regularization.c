#include <stdio.h>
#include "linear_model.h"
#include "matrix.h"
#include "util.h"
#include "regularization.h"
#include "regularization_gradient.h"
#include "metrics.h"

int main()
{
	// setup model and its functions
	LinearModel* lm = gmf_model_linear_init();
	gmf_model_linear_set_activation(&lm, &gmf_activation_identity);
	gmf_model_linear_set_loss(&lm, &gmf_loss_squared);
	gmf_model_linear_set_loss_gradient(&lm, &gmf_loss_gradient_squared);

	// setup model parameters
	gmf_model_linear_set_iterations(&lm, 100000);
	gmf_model_linear_set_learning_rate(&lm, 0.001f);
	gmf_model_linear_set_model_type(&lm, CLASSIC);

	// generate fake random data
	// NOTE: this is complete noise and no meaningful relationships
	// will be found, but just for demonstration
	Matrix* X = NULL;
	mat_init(&X, 10, 4);
	mat_random(&X, 0.0f, 10.0f);

	Matrix* Y = NULL;
	mat_init(&Y, 10, 1);
	mat_random(&Y, 2.0f, 15.0f);

	// it's recommended to add a bias term if one isn't present already
	// this adds a column vector of 1s to represent the bias
	gmf_util_add_bias(&X);

	gmf_model_linear_fit(&lm, X, Y, true);
	
	printf("\n\nACTUALS:\n");
	mat_print(Y);

	printf("\n\nPREDICTED (no regularization):\n");
	Matrix* preds = gmf_model_linear_predict(lm, X);
	mat_print(preds);
	printf("\nMean Absolute Error: %f\n", gmf_metrics_mae(Y, preds, NULL));

	// add L2 regularization with lambda = 0.5 
	float reg_params[1] = {0.5f};
	gmf_model_linear_set_regularization(&lm, &gmf_regularization_L2);
	gmf_model_linear_set_regularization_gradient(&lm, &gmf_regularization_gradient_L2);
	gmf_model_linear_set_regularization_params(&lm, reg_params, 1); // declaring we are passing 1 parameter, 0.5f

	gmf_model_linear_fit(&lm, X, Y, true);
	gmf_model_linear_predict_inplace(lm, X, &preds);
	printf("\n\nPREDICTED (regularization):\n");
	mat_print(preds);
	printf("\nMean Absolute Error: %f\n", gmf_metrics_mae(Y, preds, NULL));

	gmf_model_linear_free(&lm);
	mat_free(&X);
	mat_free(&Y);
	mat_free(&preds);

	return 0;
}
