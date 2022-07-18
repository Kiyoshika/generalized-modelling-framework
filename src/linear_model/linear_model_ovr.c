#include "linear_model.h"
#include "linear_model_ovr.h"
#include "matrix.h"
#include "gmf_util.h"

static void err(const char* msg)
{
	printf("%s\n", msg);
	exit(-1);
}

static size_t __factorial(size_t n)
{
	size_t result = n;
	for (n = result - 1; n > 0; --n)
		result *= n;
	return result;
}

static size_t __calculate_required_models(const size_t n_classes)
{
	if (n_classes < 2)
		err("Must have at least two classes to use OVR model.\n");
	// n choose r
	// n_classes choose 2
	// n_classes! / (2*(n_classes - 2)!)
	return __factorial(n_classes) / (2 * __factorial(n_classes - 2)); 
}

static void __compute_class_pairs(
		size_t (**class_pairs)[2], 
		const size_t n_models,
		const size_t n_classes)
{
	

	size_t model = 0;
	size_t x = 0, y = 1;
	while (model < n_models)
	{
		(*class_pairs)[model][0] = x;
		(*class_pairs)[model][1] = y;
		y++;
		if (y > n_classes - 1)
		{
			x++;
			y = x + 1;
		}
		model++;
	}
}

void gmf_model_linear_ovr_init_inplace(
	LinearModelOVR** lm,
	const size_t n_classes)
{
	void* alloc = malloc(sizeof(LinearModelOVR));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR.");
	*lm = alloc;

	// calculate total number of models requred given n_classes
	size_t n_models = __calculate_required_models(n_classes);
	(*lm)->n_models = n_models;

	// calculate all class combinations
	alloc = malloc(n_models * sizeof(*(*lm)->class_pairs));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR.");
	(*lm)->class_pairs = alloc;
	__compute_class_pairs(&(*lm)->class_pairs, n_models, n_classes);

	// allocate memory for all linear models
	alloc = malloc(n_models * sizeof(LinearModel));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR.");
	(*lm)->models = alloc;

	// initialize all models with default parameters
	for (size_t n = 0; n < n_models; ++n)
	{
		(*lm)->models[n] = gmf_model_linear_init();
		(*lm)->models[n]->X = NULL;
		(*lm)->models[n]->W = NULL;
	}
}

LinearModelOVR* gmf_model_linear_ovr_init(const size_t n_classes)
{
	LinearModelOVR* lm = NULL;
	gmf_model_linear_ovr_init_inplace(&lm, n_classes);

	return lm;
}

/*
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
*/

void gmf_model_linear_ovr_free(
	LinearModelOVR** lm)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		gmf_model_linear_free(&(*lm)->models[m]);

	free((*lm)->models);
	(*lm)->models = NULL;

	free((*lm)->class_pairs);
	(*lm)->class_pairs = NULL;

	free(*lm);
	*lm = NULL;

}
