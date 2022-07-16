#include <stdio.h>
#include "linear_model.h"
#include "matrix.h"
#include "util.h"

int main()
{
	// setup model and its functions
	LinearModel* lm = gmf_model_linear_init();
	lm->activation = &gmf_activation_identity;
	lm->loss = &gmf_loss_squared;
	lm->loss_gradient = &gmf_loss_gradient_squared;

	// setup model parameters
	lm->params->n_iterations = 100000;
	lm->params->learning_rate = 0.001f;

	Matrix* X = NULL;
	mat_init(&X, 10, 4);
	mat_random(&X, 0.0f, 10.0f);

	Matrix* Y = NULL;
	mat_init(&Y, 10, 1);
	mat_random(&Y, 2.0f, 15.0f);

	Matrix* X_test = mat_copy(X);
	gmf_util_add_bias(&X_test);

	gmf_model_linear_fit(&lm, X, Y);
	
	printf("\n\nACTUALS:\n");
	mat_print(Y);

	printf("\n\nPREDICTED:\n");
	Matrix* preds = gmf_model_linear_predict(lm, X_test);
	// can also use gmf_model_linear_predict_inplace(lm, X_test, &preds)
	// if preds is already allocated
	mat_print(preds);

	gmf_model_linear_free(&lm);
	mat_free(&X);
	mat_free(&X_test);
	mat_free(&X_test);
	mat_free(&Y);
	mat_free(&preds);

	return 0;
}
