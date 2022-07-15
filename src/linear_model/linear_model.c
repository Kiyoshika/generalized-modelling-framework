#include "linear_model.h"
#include "matrix.h"

static void err(const char* msg)
{
	printf("%s\n", msg);
	exit(-1);
}

void gmf_model_linear_init_inplace(
	LinearModel** lm,
	const LinearModelParams* params)
{
	void* alloc = malloc(sizeof(LinearModel));
	if (!alloc)
		err("Couldn't allocate memory for LinearModel.");
	*lm = alloc;
	(*lm)->params = params; 

	// by default we'll init X and W to NULL since they aren't set until fit() is called
	(*lm)->X = NULL;
	(*lm)->W = NULL;
}

// set default parameters
void __default_params(LinearModelParams* params)
{
	params->n_iterations = 100;
	params->learning_rate = 0.001f;
}

LinearModel* gmf_model_linear_init()
{
	LinearModel* lm = NULL;
	LinearModelParams* params = NULL;
	void* alloc = malloc(sizeof(LinearModelParams));
	if (!alloc)
		err("Couldn't allocate memory for LinearModeParams.");
	params = alloc;
	__default_params(params);
	gmf_model_linear_init_inplace(&lm, params);
	return lm;
}

static void __init_X(
		LinearModel** lm,
		const Matrix* X)
{
	// if fit is called multiple times, need to check
	// if X is already initialized
	if ((*lm)->X)
		mat_free(&(*lm)->X);

	// create a copy of X and add the bias term
	Matrix* X_cpy = NULL;
	mat_init(&X_cpy, X->n_rows, X->n_columns + 1);
	for (size_t r = 0; r < X_cpy->n_rows; ++r)
	{
		mat_set(&X_cpy, r, 0, 1.0f);
		for (size_t c = 1; c < X_cpy->n_columns; ++c)
			mat_set(&X_cpy, r, c, mat_at(X, r, c - 1));
	}
	(*lm)->X = X_cpy;
}

static void __init_W(LinearModel** lm)
{
	// if fit is called multiple times, need to check
	// if W is already initialized
	if ((*lm)->W)
		mat_free(&(*lm)->W);

	// initialize random weights in [-1, 1]
	(*lm)->W = NULL; 
	mat_init(&(*lm)->W, (*lm)->X->n_columns, 1);
	mat_random(&(*lm)->W, -1.0f, 1.0f);
}

// linear must must have:
// * activation function
// * loss function
// * loss gradient
void __check_functions(const LinearModel* lm)
{
	if (!lm->activation)
		err("LinearModel must have activation function. See gmf_activation_...");
	if (!lm->loss)
		err("LinearModel must have loss function. See gmf_loss_...");
	if (!lm->loss_gradient)
		err("LinearModel must have loss gradient. See gmf_loss_gradient...");
}

void gmf_model_linear_fit(
		LinearModel** lm,
		const Matrix* X,
		const Matrix* Y)
{
	__check_functions(*lm);
	__init_X(lm, X);
	__init_W(lm);

	// TODO: determine whether CLASSIC, BATCH or STOCHASTIC was chosen
	// for now, I will just implement the classic version

	Matrix* Yhat = NULL;
	mat_init(&Yhat, (*lm)->X->n_rows, 1);

	Matrix* loss_grad = NULL;
	mat_init(&loss_grad, (*lm)->X->n_columns, 1);

	float initial_loss = 0.0f;

	printf("max iter: %zu\n", (*lm)->params->n_iterations);

	// begin training
	for (size_t iter = 0; iter < (*lm)->params->n_iterations; ++iter)
	{
		// get linear combination of data and weights
		mat_multiply_inplace((*lm)->X, (*lm)->W, &Yhat);
		
		// apply activation
		(*lm)->activation(&Yhat);

		// only print loss 10 times for any given number
		// of iterations
		if (iter % (size_t)((float)(*lm)->params->n_iterations / 10.0f) == 0)
		{
			float loss = (*lm)->loss(Y, Yhat);
			if (iter == 0)
				initial_loss = loss;
			else if (iter > 0 && loss > 10 * initial_loss)
			{
				printf("WARNING: loss blew up. Consider lowering your learning rate.\n");
				break;
			}

			printf("Loss at iteration %zu: %f\n", iter, loss);
		}	


		(*lm)->loss_gradient(Y, Yhat, (*lm)->X, &loss_grad);

		// update weights
		mat_multiply_s(&loss_grad, (*lm)->params->learning_rate);
		mat_subtract_e(&(*lm)->W, loss_grad);
	}	

	mat_free(&Yhat);
	mat_free(&loss_grad);

}

void gmf_model_linear_free(
	LinearModel** lm)
{
	if ((*lm)->X)
		mat_free(&(*lm)->X);
	if ((*lm)->W)
		mat_free(&(*lm)->W);
	free((*lm)->params);
	(*lm)->params = NULL;
	free(*lm);
	*lm = NULL;
}
