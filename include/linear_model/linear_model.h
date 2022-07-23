#ifndef LINEAR_MODEL_H 
#define LINEAR_MODEL_H 

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "activations.h"
#include "losses.h"
#include "loss_gradients.h"
#include "gmf_util.h"

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
	size_t n_iterations; 
	float learning_rate;
	float early_stop_threshold;
	size_t early_stop_iterations;
	LinearModelType model_type; // type of optimization
	size_t batch_size;
	float* class_weights;
	size_t* class_pair;
	float* regularization_params;
} LinearModelParams;

typedef struct LinearModel
{
	LinearModelParams* params; 
	Matrix* W; // weights (coefficients of the model) - set during fit()
	void (*activation)(Matrix**);
	float (*loss)(const Matrix*, const Matrix*, const LinearModel*); 
	void (*loss_gradient)(const Matrix*, const Matrix*, const Matrix*, const LinearModel*, Matrix**);
	float (*regularization)(const float*, const Matrix*);
	float (*regularization_gradient)(const float*, const Matrix*);
} LinearModel;

// initialize new linear model by passing address of (NULL) pointer 
void gmf_model_linear_init_inplace(
	LinearModel** lm,
	const LinearModelParams* params);

// initalize new linear model and return a pointer
LinearModel* gmf_model_linear_init();

// train model given actuals: X - (r, c) matrix. Y - (r, 1) matrix.
// NOTE: a bias term is automatically added to X and stored internally so it becomes an (r+1, c) matrix.
// Original matrix is *UNTOUCHED* and still must be free'd separately.
void gmf_model_linear_fit(
	LinearModel** lm,
	const Matrix* X,
	const Matrix* Y,
	const bool verbose);

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

// set n_iterations parameter
void gmf_model_linear_set_iterations(
	LinearModel** lm,
	const size_t n_iterations);

// set learning_rate parameter
void gmf_model_linear_set_learning_rate(
	LinearModel** lm,
	const float learning_rate);

// set early_stop_threshold parameter
void gmf_model_linear_set_early_stop_threshold(
	LinearModel** lm,
	const float early_stop_threshold);

// set early_stop_iterations parameter
void gmf_model_linear_set_early_stop_iterations(
	LinearModel** lm,
	const size_t early_stop_iterations);

// set model_type parameter
void gmf_model_linear_set_model_type(
	LinearModel** lm,
	const LinearModelType model_type);

// set batch_size parameter
void gmf_model_linear_set_batch_size(
	LinearModel** lm,
	const size_t batch_size);

// pass an array of regularization params and store a copy
void gmf_model_linear_set_regularization_params(
	LinearModel** lm,
	const float* regularization_params,
	const size_t n);

// set activation function
void gmf_model_linear_set_activation(
	LinearModel** lm,
	void (*activation)(Matrix**));

// set loss function
void gmf_model_linear_set_loss(
	LinearModel** lm,
	float (*loss)(const Matrix*, const Matrix*, const LinearModel*));

// set loss gradient function
void gmf_model_linear_set_loss_gradient(
	LinearModel** lm,
	void (*loss_gradient)(const Matrix*, const Matrix*, const Matrix*J, const LinearModel*, Matrix**));

// set regularization function
void gmf_model_linear_set_regularization(
	LinearModel** lm,
	float (*regularization)(const float*, const Matrix*));

// set regularization gradient function
void gmf_model_linear_set_regularization_gradient(
	LinearModel** lm,
	float (*regularization_gradient)(const float*, const Matrix*));

#endif
