#include "linear_model.h"
#include "linear_model_ovr.h"
#include "matrix.h"
#include "vector.h"
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

static bool __filter_class(const Vector* row, float* args)
{
	return fabsf(vec_at(row, 0) - args[0]) < 0.001f || fabsf(vec_at(row, 0) - args[1]) < 0.001f;
}

void gmf_model_linear_ovr_fit(
		LinearModelOVR** lm,
		const Matrix* X,
		const Matrix* Y)
{
	// iterate over different class pairs, filter
	// the correct data, convert to [0, 1] and fit a regular linear model
	
	size_t* filtered_idx = NULL;
	for (size_t model = 0; model < (*lm)->n_models; ++model)
	{
		size_t* class_pair = (*lm)->class_pairs[model];
		float class_pair_f[2] = { (float)class_pair[0], (float)class_pair[1] };
		Matrix* Y_filtered = mat_filter(Y, &__filter_class, class_pair_f, &filtered_idx);
		Matrix* X_filtered = mat_subset_idx(X, filtered_idx, Y_filtered->n_rows);

		free(filtered_idx);
		filtered_idx = NULL;

		for (size_t r = 0; r < Y_filtered->n_rows; ++r)
		{
			if (fabsf(mat_at(Y_filtered, r, 0) - class_pair_f[0]) < 0.001f)
				mat_set(&Y_filtered, r, 0, 0.0f);
			else
				mat_set(&Y_filtered, r, 0, 1.0f);
		}

		gmf_model_linear_fit(&(*lm)->models[model], X_filtered, Y_filtered);

		mat_free(&X_filtered);
		mat_free(&Y_filtered);
	}
}
/*
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
