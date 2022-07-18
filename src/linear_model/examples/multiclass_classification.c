#include "linear_model_ovr.h"
#include <stdio.h>

int main()
{
	LinearModelOVR* ovr_model = gmf_model_linear_ovr_init(5);
	for (size_t model = 0; model < ovr_model->n_models; ++model)
		printf("%d, %d\n", ovr_model->class_pairs[model][0], ovr_model->class_pairs[model][1]);

	printf("%d\n", ovr_model->models[3]->params->n_iterations);

	gmf_model_linear_ovr_free(&ovr_model);
}
