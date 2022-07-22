/*
 * EXAMPLE: least squares regression
 *
 * This example highlights a couple of different features:
 * - setting activation, loss and gradient functions
 * - setting different model parameters
 * - adding a bias term to a new test data set (incase you don't have one in your dataset)
 * - changing the model optimization type (CLASSIC vs STOCHASTIC vs BATCH)
 * - making new predictions and using predict_inplace for multiple predictions
 */

#include <stdio.h>
#include "linear_model.h"
#include "matrix.h"
#include "util.h"
#include "regularization.h"
#include "regularization_gradient.h"

int main()
{
	// setup model and its functions
	LinearModel* lm = gmf_model_linear_init();
	lm->activation = &gmf_activation_identity;
	lm->loss = &gmf_loss_squared;
	lm->loss_gradient = &gmf_loss_gradient_squared;

	// setup model parameters
	gmf_model_linear_set_iterations(&lm, 10000);
	gmf_model_linear_set_learning_rate(&lm, 0.001f);
	gmf_model_linear_set_model_type(&lm, CLASSIC);

	// by default, early stop will happen after 10% of consecutive n_iterations
	// has no improvement on loss function. If you want to disable early stopping,
	// set early_stop_iterations = n_iterations
	gmf_model_linear_set_early_stop_iterations(&lm, lm->params->n_iterations);

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

	gmf_model_linear_fit(&lm, X, Y);
	
	printf("\n\nACTUALS:\n");
	mat_print(Y);

	printf("\n\nCLASSIC PREDICTED:\n");
	Matrix* preds = gmf_model_linear_predict(lm, X);
	mat_print(preds);

	gmf_model_linear_set_model_type(&lm, BATCH);
	gmf_model_linear_set_batch_size(&lm, 5); // by default, batch_size is 25% of original data size

	gmf_model_linear_fit(&lm, X, Y);
	printf("\n\nBATCH PREDICTED\n");
	// preds is already allocated, so we can use inplace here
	gmf_model_linear_predict_inplace(lm, X, &preds);
	mat_print(preds);

	gmf_model_linear_set_model_type(&lm, STOCHASTIC);

	gmf_model_linear_fit(&lm, X, Y);
	printf("\n\nSTOCHASTIC PREDICTED:\n");
	// preds is already allocated, so we can use inplace here
	gmf_model_linear_predict_inplace(lm, X, &preds); 
	mat_print(preds);


	gmf_model_linear_free(&lm);
	mat_free(&X);
	mat_free(&Y);
	mat_free(&preds);

	return 0;
}
