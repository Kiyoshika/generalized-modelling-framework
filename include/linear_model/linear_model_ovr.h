#ifndef LINEAR_MODEL_OVR_H 
#define LINEAR_MODEL_OVR_H 

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "activations.h"
#include "losses.h"
#include "loss_gradients.h"
#include "gmf_util.h"
#include "linear_model.h"

// forward declaration
typedef struct Matrix Matrix;

typedef struct LinearModelOVR
{
	LinearModel** models; // OVR is a collection of models with different paired labels
	size_t n_models;
	size_t n_classes;
	size_t (*class_pairs)[2]; // pairs of [0, 1], [0, 2], [1, 2] etc. class labels per linear model
	float* class_weights; // weights for each class in order [0, 1, 2, ...]. Higher the value, more importance is given.
} LinearModelOVR;

// initialize new linear model by passing address of (NULL) pointer 
void gmf_model_linear_ovr_init_inplace(
	LinearModelOVR** lm,
	const size_t n_classes,
	const float* class_weights);

// initalize new linear model and return a pointer
LinearModelOVR* gmf_model_linear_ovr_init(
		const size_t n_classes,
		const float* class_weights);

// train model given actuals: X - (r, c) matrix. Y - (r, 1) matrix.
// NOTE: a bias term is automatically added to X and stored internally so it becomes an (r+1, c) matrix.
// Original matrix is *UNTOUCHED* and still must be free'd separately.
void gmf_model_linear_ovr_fit(
	LinearModelOVR** lm,
	const Matrix* X,
	const Matrix* Y);

// Take data X and make predictions using linear model.
// Predictions are allocated and returned as new matrix.
Matrix* gmf_model_linear_ovr_predict(
	const LinearModelOVR* lm,
	const Matrix* X);

// Take data X and make predictions using linear model.
// Predictions are stored into Yhat and is assumed to be allocated to the correct size beforehand.
void gmf_model_linear_ovr_predict_inplace(
	const LinearModelOVR* lm,
	const Matrix* X,
	Matrix** Yhat);

// cleanup memory
void gmf_model_linear_ovr_free(
	LinearModelOVR** lm);

// set n_iterations parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_iterations(
		LinearModelOVR** lm,
		size_t n_iterations);

// set learning_rate parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_learning_rate(
		LinearModelOVR** lm,
		float learning_rate);

// set early_stop_threshold parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_early_stop_threshold(
		LinearModelOVR** lm,
		float early_stop_threshold);

// set early_stop_iterations parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_early_stop_iterations(
		LinearModelOVR** lm,
		size_t early_stop_iterations);

// set model_type parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_model_type(
		LinearModelOVR** lm,
		LinearModelType model_type);

// set batch_size parameter for all submodels in OVR model
void gmf_model_linear_ovr_set_batch_size(
		LinearModelOVR** lm,
		size_t batch_size);

// set activation function for all submodels in OVR model
void gmf_model_linear_ovr_set_activation(
		LinearModelOVR** lm,
		void (*activation)(Matrix**));

// set loss function for all submodels in OVR model
void gmf_model_linear_ovr_set_loss(
		LinearModelOVR** lm,
		float (*loss)(const Matrix*, const Matrix*));

// set loss gradient functions for all submodels in OVR model
void gmf_model_linear_ovr_set_loss_gradient(
		LinearModelOVR** lm,
		void (*loss_gradient)(const Matrix*, const Matrix*, const Matrix*, const size_t*, const float*, Matrix**));
#endif
