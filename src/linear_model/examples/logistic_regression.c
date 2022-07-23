/*
 * EXAMPLE: logistic regression
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

int main()
{
	// setup model and its functions
	LinearModel* lm = gmf_model_linear_init();
	gmf_model_linear_set_activation(&lm, &gmf_activation_sigmoid);
	gmf_model_linear_set_loss(&lm, &gmf_loss_cross_entropy);
	gmf_model_linear_set_loss_gradient(&lm, &gmf_loss_gradient_cross_entropy);

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
	mat_random(&Y, 0.0f, 1.0f);

	// convert Y to hard 0 & 1
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		if (mat_at(Y, r, 0) < 0.5f)
			mat_set(&Y, r, 0, 0.0f);
		else
			mat_set(&Y, r, 0, 1.0f);
	}

	gmf_util_add_bias(&X);

	gmf_model_linear_fit(&lm, X, Y, true);
	
	printf("\n\nACTUALS:\n");
	mat_print(Y);

	printf("\n\nPREDICTED:\n");
	Matrix* preds = gmf_model_linear_predict(lm, X);
	mat_print(preds);

	gmf_model_linear_free(&lm);
	mat_free(&X);
	mat_free(&Y);
	mat_free(&preds);

	return 0;
}
