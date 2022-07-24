#include "losses.h"
#include "matrix.h"
#include "linear_model.h"

static float __squared_loss(
		float y, 
		float yhat,
		const LinearModel* lm)
{
	return (y - yhat) * (y - yhat);
}

static float __cross_entropy_loss(
		float y, 
		float yhat,
		const LinearModel* lm)
{
	// constrain yhat between [0, 1]
	yhat = yhat < 0.01f ? 0.01f : yhat;
	yhat = yhat > 0.99f ? 0.99f : yhat;
	return -y * logf(yhat) - (1 - y) * logf(1 - yhat);
}

static float __absolute_loss(
		float y, 
		float yhat,
		const LinearModel* lm)
{
	return fabsf(y - yhat);
}

static float __hinge_loss(
		float y, 
		float yhat,
		const LinearModel* lm)
{
	// hinge loss is specifically defined foor {-1, 1}
	// so we convert 0 to -1 for the sake of this function only
	if (fabsf(y - 0.0f) < 0.0001f)
		y = -1.0f;
	if (fabsf(yhat - 0.0f) < 0.0001f)
		yhat = -1.0f;
	
	// even though it is technically impossible to go
	// below zero, I match the original definition
	// anyways
	return 1 - y * yhat < 0.0f ? 0.0f : 1 - y * yhat;
}

static float __huber_loss(
		float y, 
		float yhat,
		const LinearModel* lm)
{
	if (fabsf(y - yhat) < lm->params->huber_delta)
		return 0.5f * (y - yhat) * (y - yhat);

	return lm->params->huber_delta * (fabsf(y - yhat) - 0.5f * lm->params->huber_delta);
}

static float __compute_loss(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm,
		float (*loss_func)(float, float, const LinearModel*))
{
	float loss = 0.0f;
	for (size_t r = 0; r < Y->n_rows; ++r)
	{
		float y = mat_at(Y, r, 0);
		float yhat = mat_at(Yhat, r, 0);
		loss += loss_func(y, yhat, lm);
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

float gmf_loss_absolute(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm)
{
	return __compute_loss(Y, Yhat, lm, &__absolute_loss);
}

float gmf_loss_hinge(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm)
{
	return __compute_loss(Y, Yhat, lm, &__hinge_loss);
}

float gmf_loss_huber(
		const Matrix* Y,
		const Matrix* Yhat,
		const LinearModel* lm)
{
	return __compute_loss(Y, Yhat, lm, &__huber_loss);
}
