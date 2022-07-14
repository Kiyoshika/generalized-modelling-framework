#include <stdio.h>
#include "linear_model.h"

int main()
{
	LinearModelParams params;
	params.n_iterations = 10;

	LinearModel* lm = gmf_model_linear_init(params);
	printf("iterations: %zu\n", lm->params.n_iterations);

	gmf_model_linear_free(&lm);
	return 0;
}
