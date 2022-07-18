#include "activations.h"
#include "matrix.h"

void gmf_activation_identity(Matrix** XW)
{
	// identity activation actuall does nothing
	// so we just return
	return;
}

void gmf_activation_sigmoid(Matrix** XW)
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
