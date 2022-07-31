#include "knn.h"
#include "matrix.h"
#include "vector.h"

static void knn_err(const char* msg)
{
	printf("%s\n", msg);
	exit(-1);
}

static void __default_params(KNN** knn)
{
	(*knn)->type = CLASSIC;
	(*knn)->params->distance = &gmf_distance_euclidean;
	(*knn)->params->n_neighbors = 3;
}

KNN* gmf_model_knn_init()
{
	void* alloc = malloc(sizeof(KNN));
	if (!alloc)
		knn_err("Couldn't allocate memory for KNN.");

	KNN* knn = alloc;

	alloc = malloc(sizeof(KNNParams));
	if (!alloc)
		knn_err("Couldn't allocate memory for KNN.");

	knn->params = alloc;

	__default_params(&knn);

	return knn;
}

void gmf_model_knn_set_type(
		KNN** knn,
		const KNNType type)
{
	(*knn)->type = type;
}

void gmf_model_knn_set_distance(
		KNN** knn,
		float (*distance)(const Vector*, const Vector*))
{
	(*knn)->params->distance = distance;
}

void gmf_model_knn_set_neighbors(
		KNN** knn,
		const size_t n_neighbors)
{
	(*knn)->params->n_neighbors = n_neighbors;
}

void gmf_model_knn_fit(
		KNN** knn,
		const Matrix* X,
		const Matrix* Y)
{
	// KNN stores copy
	(*knn)->X = mat_copy(X);
	(*knn)->Y = mat_copy(Y);
}

typedef struct distance_pair
{
	float distance;
	size_t idx;
} distance_pair;

int distance_comparator(const void* dp1, const void* dp2)
{
	distance_pair _dp1 = *(distance_pair*)dp1;
	distance_pair _dp2 = *(distance_pair*)dp2;
	
	return _dp1.distance < _dp2.distance;
}

Matrix* gmf_model_knn_predict(
		const KNN* knn,
		const Matrix* X)
{
	Vector* row_vec = NULL;
	Vector* test_row = NULL;
	distance_pair* distance_pairs = NULL;

	void* alloc = malloc(knn->X->n_rows * sizeof(distance_pair));
	if (!alloc)
		knn_err("Couldn't allocate memory to store distances for KNN.");

	distance_pairs = alloc;

	Matrix* predicted = NULL;
	mat_init(&predicted, X->n_rows, 1);

	for (size_t r = 0; r < X->n_rows; ++r)
	{
		row_vec = mat_get_row(X, r);	
		switch (knn->type)
		{
			case CLASSIC:
				#include "knn_classic.c"
				break;
			case KDTree:
				// not implemented yet
				printf("KDTree method not implemented yet.");
				exit(-1);
				break;
		}	

		vec_free(&row_vec);

		qsort(distance_pairs, knn->X->n_rows, sizeof(distance_pair), &distance_comparator);

		float estimate = 0.0f;
		for (size_t k = 0; k < knn->params->n_neighbors; ++k)
			estimate += mat_at(knn->Y, distance_pairs[k].idx, 0);

		mat_set(&predicted, r, 0, estimate/(float)knn->params->n_neighbors);
	}

	free(distance_pairs);
	
	return predicted;
}

void gmf_model_knn_free(KNN** knn)
{
	free((*knn)->params);
	(*knn)->params = NULL;

	mat_free(&(*knn)->X);
	mat_free(&(*knn)->Y);

	free(*knn);
	*knn = NULL;
}
