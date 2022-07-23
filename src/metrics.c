#include "metrics.h"
#include "matrix.h"

static void __check_row_count(
		const Matrix* Y, 
		const Matrix* Yhat, 
		const char* metric)
{
	if (Y->n_rows != Yhat->n_rows)
	{
		printf("Mismatch of rows when computing %s.\n", metric);
		exit(-1);
	}
}

float gmf_metrics_mae(
		const Matrix* Y, 
		const Matrix* Yhat,
		void* params)
{
	__check_row_count(Y, Yhat, "mean absolute error");

	float sum = 0.0f;
	for (size_t r = 0; r < Y->n_rows; ++r)
		sum += fabsf(mat_at(Y, r, 0) - mat_at(Yhat, r, 0));

	return sum / (float)Y->n_rows;
}

float gmf_metrics_mse(
		const Matrix* Y, 
		const Matrix* Yhat, 
		void* params)
{
	__check_row_count(Y, Yhat, "mean squared error");

	float sum = 0.0f;
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float diff = mat_at(Y, r, 0) - mat_at(Yhat, r, 0);
		sum += diff * diff;
	}

	return sum / (float)Y->n_rows;
}

static void* __try_calloc(size_t n_memb, size_t size)
{
	void* alloc = calloc(n_memb, size);
	if (!alloc)
	{
		printf("Couldn't allocate memory when constructing confusion matrix.\n");
		exit(-1);
	}

	return alloc;
}

float gmf_metrics_confusion_matrix(
		const Matrix* Y, 
		const Matrix* Yhat, 
		void* params)
{
	size_t TP = 0;
	size_t FP = 0;
	size_t FN = 0;
	size_t n_classes = *(size_t*)params;
	size_t* class_count = __try_calloc(n_classes, sizeof(size_t));
	size_t* predicted_labels = __try_calloc(n_classes, sizeof(size_t));
	size_t* actual_labels = __try_calloc(n_classes, sizeof(size_t));


	free(class_count);
	free(predicted_labels);
	free(actual_labels);
}
