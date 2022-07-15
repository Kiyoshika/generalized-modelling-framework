#include "losses.h"
#include "matrix.h"

float gmf_loss_squared(
		const Matrix* Y,
		const Matrix* Yhat)
{
	float loss = 0.0f;
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);
		loss += (y - yhat) * (y - yhat);
	}

	return loss;
}
