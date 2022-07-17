#include "linear_model.h"
#include "matrix.h"
#include "gmf_util.h"

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
	params->early_stop_threshold = 0.0001f;
	params->early_stop_iterations = 0;
	params->model_type = CLASSIC;
	params->batch_size = 0;
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

// check if loss hasn't really improved for N iterations
bool __check_loss_tolerance(
		const float loss, 
		const float previous_loss, 
		const float tolerance,
		size_t* tolerance_counter,
		const size_t early_stop_iterations)
{
	if (fabsf(loss - previous_loss) < tolerance)
		(*tolerance_counter)++;
	else
		*tolerance_counter = 0;

	if (*tolerance_counter >= early_stop_iterations)
	{
		printf("NOTE: no improvement in loss after %zu consecutive iterations. Stopped early.\n", early_stop_iterations);
		return true;
	}

	return false;
}

void gmf_model_linear_fit(
		LinearModel** lm,
		const Matrix* X,
		const Matrix* Y)
{
	__check_functions(*lm);
	__init_X(lm, X);
	__init_W(lm);

	// set default batch size if one wasn't set (default of 25% original data size)
	if ((*lm)->params->model_type == BATCH && (*lm)->params->batch_size == 0)
		(*lm)->params->batch_size = (*lm)->X->n_rows / 4;
	// set default early_stop_iterations if one wasn't set (default is 10% original iterations)
	if ((*lm)->params->early_stop_iterations == 0)
		(*lm)->params->early_stop_iterations = (*lm)->params->n_iterations / 10;

	Matrix* loss_grad = NULL;
	mat_init(&loss_grad, (*lm)->X->n_columns, 1);

	float initial_loss = 0.0f;
	float previous_loss = 0.0f;
	size_t tolerance_counter = 0;
	bool stop_early = false;

	printf("max iter: %zu\n", (*lm)->params->n_iterations);

	// begin training
	switch((*lm)->params->model_type)
	{
		case CLASSIC:
			#include "./model_types/linear_model_classic.c"
			break;
		case BATCH:
			#include "./model_types/linear_model_batch.c"
			break;
		case STOCHASTIC:
			#include "./model_types/linear_model_stochastic.c"
			break;
	}

	// only display this message if early stopping is not disabled
	if (!stop_early && (*lm)->params->early_stop_iterations < (*lm)->params->n_iterations)
		printf("WARNING: model may not have converged. Consider increasing iterations or learning rate.\n");

	mat_free(&loss_grad);

}

Matrix* gmf_model_linear_predict(
		const LinearModel* lm,
		const Matrix* X)
{
	Matrix* Yhat= mat_multiply(X, lm->W);
	lm->activation(&Yhat);
	
	return Yhat;
}

void gmf_model_linear_predict_inplace(
		const LinearModel* lm,
		const Matrix* X,
		Matrix** Yhat)
{
	mat_multiply_inplace(X, lm->W, Yhat);
	lm->activation(Yhat);
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
