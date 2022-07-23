#ifndef METRICS_H
#define METRICS_H

#include <math.h>

// forward declaration
typedef struct Matrix Matrix;

// mean aboslute error: f(y, yhat) = mean(|y - yhat|)
// no additional params are used in this function
float gmf_metrics_mae(
		const Matrix* Y, 
		const Matrix* Yhat,
		void* params);

// mean squared error: f(y, yhat) = mean((y - yhat)^2)
// no additional params are used in this function
float gmf_metrics_mse(
		const Matrix* Y, 
		const Matrix* Yhat,
		void* params);

// confusion matrix - prints a confusion matrix with the
// rows being actuals and columns being predicted values
// from 0 to N classes. returned float is the weighted F1 score:
// TP / (TP + 0.5(FP + FN)) weighted by the presence of each class
//
// a size_t is passed to params to represent the # of classes
float gmf_metrics_confusion_matrix(
		const Matrix* Y, 
		const Matrix* Yhat,
		void* params);

#endif
