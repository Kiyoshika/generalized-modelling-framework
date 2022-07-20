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
	const size_t n_classes,
	const float* class_weights)
{
	void* alloc = malloc(sizeof(LinearModelOVR));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR.");
	*lm = alloc;
	(*lm)->n_classes = n_classes;

	// calculate total number of models requred given n_classes
	size_t n_models = __calculate_required_models(n_classes);
	(*lm)->n_models = n_models;

	if (!class_weights)
		(*lm)->class_weights = NULL; // this is calculated at fit() time
	else
	{
		// copy contents of class weights to not take ownership of pointer
		alloc = calloc(n_classes, sizeof(float));
		if (!alloc)
			err("Couldn't allocate memory for LinearModelOVR.");
		(*lm)->class_weights = alloc;

		for (size_t i = 0; i < n_classes; ++i)
			(*lm)->class_weights[i] = class_weights[i];
	}

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

LinearModelOVR* gmf_model_linear_ovr_init(
		const size_t n_classes,
		const float* class_weights)
{
	LinearModelOVR* lm = NULL;
	gmf_model_linear_ovr_init_inplace(&lm, n_classes, class_weights);

	return lm;
}

static bool __filter_class(const Vector* row, float* args)
{
	return fabsf(vec_at(row, 0) - args[0]) < 0.001f || fabsf(vec_at(row, 0) - args[1]) < 0.001f;
}

static float* __compute_class_weights(const Matrix* Y, const size_t n_classes)
{
	size_t* class_counts = NULL;
	void* alloc = calloc(n_classes, sizeof(size_t));
	if (!alloc)
		err("Couldn't allocate memory for computing class weights.");
	class_counts = alloc;

	for (size_t r = 0; r < Y->n_rows; ++r)
		class_counts[(size_t)mat_at(Y, r, 0)]++;

	float* class_weights = NULL;
	alloc = calloc(n_classes, sizeof(float));
	if (!alloc)
		err("Couldn't allocate memory for computing class weights.");
	class_weights = alloc;

	// class weight = N / (n_classes * class_count)
	for (size_t c = 0; c < n_classes; ++c)
		class_weights[c] = (float)Y->n_rows / (float)(n_classes * class_counts[c]);

	return class_weights;
}

void gmf_model_linear_ovr_fit(
		LinearModelOVR** lm,
		const Matrix* X,
		const Matrix* Y)
{
	
	// compute class weights if they aren't specified
	if ((*lm)->class_weights == NULL)
		(*lm)->class_weights = __compute_class_weights(Y, (*lm)->n_classes);

	// models will share the same pointer to save memory
	for (size_t m = 0; m < (*lm)->n_models; ++m)
	{
		(*lm)->models[m]->params->class_weights = (*lm)->class_weights;
		(*lm)->models[m]->params->class_pair = (*lm)->class_pairs[m];
	}

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

static float __set_classes(float x, float* args)
{
	if (x < 0.5f)
		return args[0];
	return args[1];
}

Matrix* gmf_model_linear_ovr_predict(
		const LinearModelOVR* lm,
		const Matrix* X)
{

	Matrix* predicted_labels = NULL;
	mat_init(&predicted_labels, lm->n_models, X->n_rows);

	for (size_t model = 0; model < lm->n_models; ++model)
	{
		Matrix* Yhat = mat_multiply(X, lm->models[model]->W);
		float class_pairs[2] = { (float)lm->class_pairs[model][0], (float)lm->class_pairs[model][1] };
		mat_apply(&Yhat, &__set_classes, class_pairs);
		for (size_t r = 0; r < Yhat->n_rows; ++r)
			mat_set(&predicted_labels, model, r, mat_at(Yhat, r, 0));
		mat_free(&Yhat);
	}

	Matrix* Yhat = NULL;
	mat_init(&Yhat, X->n_rows, 1);

	void* alloc = calloc(lm->n_classes, sizeof(float));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR predictions.");
	float* class_lookup_table = alloc;
	for (size_t r = 0; r < X->n_rows; ++r)
	{
		// reset lookup table
		for (size_t i = 0; i < lm->n_classes; ++i)
			class_lookup_table[i] = 0.0f;

		for (size_t m = 0; m < lm->n_models; ++m)
			class_lookup_table[(size_t)mat_at(predicted_labels, m, r)] += 1.0f;

		float frequent_class = -1.0f;
		for (size_t c = 0; c < lm->n_classes; ++c)
			if (class_lookup_table[c] > class_lookup_table[(size_t)frequent_class])
				frequent_class = (float)c;

		mat_set(&Yhat, r, 0, frequent_class);
	}

	// CLEANUP
	mat_free(&predicted_labels);
	free(class_lookup_table);
	
	return Yhat;
}

void gmf_model_linear_ovr_predict_inplace(
		const LinearModelOVR* lm,
		const Matrix* X,
		Matrix** Yhat)
{
	Matrix* predicted_labels = NULL;
	mat_init(&predicted_labels, lm->n_models, X->n_rows);

	for (size_t model = 0; model < lm->n_models; ++model)
	{
		Matrix* Yhat_temp = mat_multiply(X, lm->models[model]->W);
		float class_pairs[2] = { (float)lm->class_pairs[model][0], (float)lm->class_pairs[model][1] };
		mat_apply(&Yhat_temp, &__set_classes, class_pairs);
		for (size_t r = 0; r < Yhat_temp->n_rows; ++r)
			mat_set(&predicted_labels, model, r, mat_at(Yhat_temp, r, 0));
		mat_free(&Yhat_temp);
	}

	void* alloc = calloc(lm->n_classes, sizeof(float));
	if (!alloc)
		err("Couldn't allocate memory for LinearModelOVR predictions.");
	float* class_lookup_table = alloc;
	for (size_t r = 0; r < X->n_rows; ++r)
	{
		// reset lookup table
		for (size_t i = 0; i < lm->n_classes; ++i)
			class_lookup_table[i] = 0.0f;

		for (size_t m = 0; m < lm->n_models; ++m)
			class_lookup_table[(size_t)mat_at(predicted_labels, m, r)] += 1.0f;

		float frequent_class = -1.0f;
		for (size_t c = 0; c < lm->n_classes; ++c)
			if (class_lookup_table[c] > class_lookup_table[(size_t)frequent_class])
				frequent_class = (float)c;

		mat_set(Yhat, r, 0, frequent_class);
	}

	// CLEANUP
	mat_free(&predicted_labels);
	free(class_lookup_table);
}

void gmf_model_linear_ovr_free(
	LinearModelOVR** lm)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		gmf_model_linear_free(&(*lm)->models[m]);

	free((*lm)->models);
	(*lm)->models = NULL;

	free((*lm)->class_pairs);
	(*lm)->class_pairs = NULL;

	free((*lm)->class_weights);
	(*lm)->class_weights = NULL;

	free(*lm);
	*lm = NULL;

}

void gmf_model_linear_ovr_set_iterations(
		LinearModelOVR** lm,
		size_t n_iterations)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->n_iterations = n_iterations;
}

void gmf_model_linear_ovr_set_learning_rate(
		LinearModelOVR** lm,
		float learning_rate)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->learning_rate = learning_rate;
}

void gmf_model_linear_ovr_set_early_stop_threshold(
		LinearModelOVR** lm,
		float early_stop_threshold)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->early_stop_threshold = early_stop_threshold;
}

void gmf_model_linear_ovr_set_early_stop_iterations(
		LinearModelOVR** lm,
		size_t early_stop_iterations)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->early_stop_iterations = early_stop_iterations;
}

void gmf_model_linear_ovr_set_model_type(
		LinearModelOVR** lm,
		LinearModelType model_type)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->model_type = model_type;
}

void gmf_model_linear_ovr_set_batch_size(
		LinearModelOVR** lm,
		size_t batch_size)
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->params->batch_size = batch_size;
}

void gmf_model_linear_ovr_set_activation(
		LinearModelOVR** lm,
		void (*activation)(Matrix**))
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->activation = activation;
}

void gmf_model_linear_ovr_set_loss(
		LinearModelOVR** lm,
		float (*loss)(const Matrix*, const Matrix*))
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->loss = loss;
}

void gmf_model_linear_ovr_set_loss_gradient(
		LinearModelOVR** lm,
		void (*loss_gradient)(const Matrix*, const Matrix*, const Matrix*, const size_t*, const float*, Matrix**))
{
	for (size_t m = 0; m < (*lm)->n_models; ++m)
		(*lm)->models[m]->loss_gradient = loss_gradient;
}
