#include "linear_model.h"
#include "matrix.h"

static void err(const char* msg)
{
	printf("%s\n", msg);
	exit(-1);
}

void gmf_model_linear_init_inplace(
	LinearModel** lm,
	const LinearModelParams params)
{
	void* alloc = malloc(sizeof(LinearModel));
	if (!alloc)
		err("Couldn't allocate memory for LinearModel.");
	*lm = alloc;
	(*lm)->params = params; // store a copy of the params to avoid taking ownership
	
	// by default we'll init X and W to NULL since they aren't set until fit() is called
	(*lm)->X = NULL;
	(*lm)->W = NULL;
}

LinearModel* gmf_model_linear_init(
	const LinearModelParams params)
{
	LinearModel* lm = NULL;
	gmf_model_linear_init_inplace(&lm, params);
	return lm;
}

void gmf_model_linear_free(
	LinearModel** lm)
{
	if ((*lm)->X)
		mat_free(&(*lm)->X);
	if ((*lm)->W)
		mat_free(&(*lm)->W);
	free(*lm);
	*lm = NULL;
}
