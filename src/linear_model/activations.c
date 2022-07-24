#include "activations.h"
#include "matrix.h"
#include "linear_model.h"

void gmf_activation_identity(
		Matrix** XW,
		const LinearModel* lm)
{
	// identity activation actuall does nothing
	// so we just return
	return;
}

void gmf_activation_sigmoid_soft(
		Matrix** XW,
		const LinearModel* lm)
{
	for (size_t r = 0; r < (*XW)->n_rows; ++r)
	{
		float actv = 1.0f / (1.0f + expf(-mat_at(*XW, r, 0)));
		// keep bounds between 0 and 1
		actv = actv < 0.0f ? 0.0f : actv;
		actv = actv > 1.0f ? 1.0f : actv;
		mat_set(XW, r, 0, actv);
	}
}

void gmf_activation_sigmoid_hard(
		Matrix** XW,
		const LinearModel* lm)
{
	gmf_activation_sigmoid_soft(XW, lm);
	for (size_t r = 0; r < (*XW)->n_rows; ++r)
	{
		if (mat_at(*XW, r, 0) > lm->params->sigmoid_threshold)
			mat_set(XW, r, 0, 1.0f);
		else
			mat_set(XW, r, 0, 0.0f);
	}
}
