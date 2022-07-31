#include "knn.h"
#include "matrix.h"
#include <stdio.h>

int main()
{
	Matrix* X = NULL;
	mat_init(&X, 10, 4);
	mat_random(&X, -5.0f, 5.0f);

	Matrix* Y = NULL;
	mat_init(&Y, 10, 1);
	mat_random(&Y, 3.0f, 10.0f);

	KNN* knn = gmf_model_knn_init();
	gmf_model_knn_fit(&knn, X, Y);

	Matrix* preds = gmf_model_knn_predict(knn, X);
	
	printf("ACTUALS:\n");
	mat_print(Y);

	printf("\n\nPREDICTED:\n");
	mat_print(preds);

	gmf_model_knn_free(&knn);
	mat_free(&X);
	mat_free(&Y);
	mat_free(&preds);

	return 0;
}
