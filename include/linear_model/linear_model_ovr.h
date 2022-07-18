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

// forward declarations
typedef struct Matrix Matrix;
typedef struct LinearModel LinearModel;
typedef struct LinearModelParams LinearModelParams;

typedef struct LinearModelOVR
{
	LinearModel** models; // OVR is a collection of models with different paired labels
	size_t n_models;
	size_t (*class_pairs)[2]; // pairs of [0, 1], [0, 2], [1, 2] etc. class labels per linear model
} LinearModelOVR;

// initialize new linear model by passing address of (NULL) pointer 
void gmf_model_linear_ovr_init_inplace(
	LinearModelOVR** lm,
	const size_t n_classes);

// initalize new linear model and return a pointer
LinearModelOVR* gmf_model_linear_ovr_init(size_t n_classes);

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

#endif
