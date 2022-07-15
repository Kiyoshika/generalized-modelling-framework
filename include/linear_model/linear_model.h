#ifndef LINEAR_MODEL_H 
#define LINEAR_MODEL_H 

#include <stdlib.h>
#include <stdint.h>

#include "activations.h"
#include "losses.h"
#include "loss_gradients.h"

// forward declaration
typedef struct Matrix Matrix;

typedef enum LinearModelType
{
	CLASSIC, // train on full data set each iteration
	BATCH, // train on sampled data set each iteration
	STOCHASTIC // train on single point each iteration
} LinearModelType;

typedef struct LinearModelParams
{
	size_t n_iterations; // total iterations to train model for
	LinearModelType model_type; // type of optimization
} LinearModelParams;

typedef struct LinearModel
{
	LinearModelParams params; // model parameters (set by user before calling init())
	Matrix* X; // internal X grabbed by fit() to add the bias term automatically
	Matrix* W; // weights (coefficients of the model) - set during fit()
	void (*activation)(Matrix**);
	float (*loss)(const Matrix*, const Matrix*); 
	void (*loss_gradient)(const Matrix*, const Matrix*, const Matrix*, Matrix**);
} LinearModel;

// initialize new linear model by passing address of (NULL) pointer 
// NOTE: a copy of params is taken incase user wants to use same struct in other models
void gmf_model_linear_init_inplace(
	LinearModel** lm,
	const LinearModelParams params);

// initalize new linear model and return a pointer
// NOTE: a copy of params is taken incase user wants to use same struct in other models
LinearModel* gmf_model_linear_init(
	const LinearModelParams params);

// train model given actuals: X - (r, c) matrix. Y - (r, 1) matrix.
// NOTE: a bias term is automatically added to X and stored internally so it becomes an (r+1, c) matrix.
// Original matrix is *UNTOUCHED* and still must be free'd separately.
void gmf_model_linear_fit(
	LinearModel** lm,
	const Matrix* X,
	const Matrix* Y);

// Take data X and make predictions using linear model.
// Predictions are allocated and returned as new matrix.
Matrix* gmf_model_linear_predict(
	const LinearModel* lm,
	const Matrix* X);

// Take data X and make predictions using linear model.
// Predictions are stored into Yhat and is assumed to be allocated to the correct size beforehand.
void gmf_model_linear_predict_inplace(
	const LinearModel* lm,
	const Matrix* X,
	Matrix** Yhat);

// cleanup memory
void gmf_model_linear_free(
	LinearModel** lm);

#endif
