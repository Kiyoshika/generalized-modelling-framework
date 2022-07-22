#include "losses.h"
#include "matrix.h"
#include "linear_model.h"

static float __squared_loss(const float y, float yhat)
{
	return (y - yhat) * (y - yhat);
}

static float __cross_entropy_loss(const float y, float yhat)
{
	// constrain yhat between [0, 1]
	yhat = yhat < 0.01f ? 0.01f : yhat;
	yhat = yhat > 0.99f ? 0.99f : yhat;
	return -y * logf(yhat) - (1 - y) * logf(1 - yhat);
}

static float __compute_loss(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm,
		float (*loss_func)(const float, float))
{
	float loss = 0.0f;
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);
		loss += loss_func(y, yhat);
	}

	float regularization = 0.0f;
	if (lm->regularization)
		regularization = lm->regularization(lm->params->regularization_params, lm->W);

	return loss + regularization;
}

float gmf_loss_squared(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm)
{
	return __compute_loss(Y, Yhat, lm, &__squared_loss);
}

float gmf_loss_cross_entropy(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm)
{
	return __compute_loss(Y, Yhat, lm, &__cross_entropy_loss);
}
