#ifndef KNN_H
#define KNN_H

#include "distances.h"
#include <stddef.h>
#include <stdlib.h>

// forward declaration
typedef struct Vector Vector;
typedef struct Matrix Matrix;

// type of KNN
typedef enum KNNType
{
	CLASSIC, // naive implementation - entire data set is used for comparisons each time
	KDTree // use KDTrees to find nearest neighbors
} KNNType;

typedef struct KNNParams
{
	float (*distance)(const Vector*, const Vector*);
	size_t n_neighbors;
} KNNParams;

typedef struct KNN
{
	KNNParams* params;
	KNNType type;
	Matrix* X;
	Matrix* Y;
} KNN;

// initialize new KNN model and return a pointer
KNN* gmf_model_knn_init();

// set KNN type
void gmf_model_knn_set_type(
		KNN** knn,
		const KNNType type);

// set distance function
void gmf_model_knn_set_distance(
		KNN** knn,
		float (*distance)(const Vector*, const Vector*));

// set n_neighbors parameter
void gmf_model_knn_set_neighbors(
		KNN** knn,
		const size_t n_neighbors);

// fit KNN model
void gmf_model_knn_fit(
		KNN** knn, 
		const Matrix* X, 
		const Matrix* Y);

// find nearest neighbors
Matrix* gmf_model_knn_predict(
		const KNN* knn, 
		const Matrix* X);

// free memory allocated by KNN model
void gmf_model_knn_free(KNN** knn);

#endif
