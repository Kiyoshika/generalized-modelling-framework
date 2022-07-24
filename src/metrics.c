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

static bool __approx_eq(float x, float y)
{
	return fabsf(x - y) < 0.0001f;
}

static float __single_class_F1(
		const Matrix* Y,
		const Matrix* Yhat,
		const float positive_label)
{
	size_t TP = 0;
	size_t FP = 0;
	size_t FN = 0;
	
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);

		if (__approx_eq(y, positive_label) && __approx_eq(yhat, positive_label))
			TP++;
		else if (!__approx_eq(y, positive_label) && __approx_eq(yhat, positive_label))
			FP++;
		else if (__approx_eq(y, positive_label) && !__approx_eq(yhat, positive_label))
			FN++;
	}

	return (float)TP / ((float)TP + 0.5f * ((float)FP + (float)FN));
}

float gmf_metrics_confusion_matrix(
		const Matrix* Y, 
		const Matrix* Yhat, 
		void* params)
{
	__check_row_count(Y, Yhat, "confusion matrix");

	size_t n_classes = *(size_t*)params;
	size_t* class_label_count;
	void* alloc = calloc(n_classes, sizeof(size_t));
	if (!alloc)
	{
		printf("Couldn't allocate memory when constructing confusion matrix.\n");
		exit(-1);
	}
	class_label_count = alloc;

	// count class labels
	for (size_t i = 0; i < Y->n_rows; ++i)
		class_label_count[(size_t)mat_at(Y, i, 0)]++;

	// calculated weighted f1 score to return
	float weighted_f1 = 0.0f;
	for (size_t c = 0; c < n_classes; ++c)
	{
		float class_f1 =  __single_class_F1(Y, Yhat, (float)c);
		weighted_f1 += class_f1 * ((float)class_label_count[c] / (float)Y->n_rows);
	}

	// construct and print the confusion matrix
	Matrix* conf_mat = NULL;
	mat_init(&conf_mat, n_classes, n_classes);
	for (size_t i = 0; i < Y->n_rows; ++i)
	{
		size_t y = (size_t)mat_at(Y, i, 0);
		size_t yhat = (size_t)mat_at(Yhat, i, 0);
		mat_set(&conf_mat, y, yhat, mat_at(conf_mat, y, yhat) + 1.0f);	
	}

	printf("\n\n[[ Confusion Matrix ]]\n\nRows: actuals\nColumns: predicted\n\n");
	printf("    ");
	for (size_t c = 0; c < conf_mat->n_columns; ++c)
		printf("%zu ", c);
	printf("\n");
	for (size_t r = 0; r < conf_mat->n_rows; ++r)
	{
		printf("%zu   ", r);
		for (size_t c = 0; c < conf_mat->n_columns; ++c)
			printf("%zu ", (size_t)mat_at(conf_mat, r, c));
		printf("\n");
	}
	mat_free(&conf_mat);

	free(class_label_count);

	return weighted_f1;
}
