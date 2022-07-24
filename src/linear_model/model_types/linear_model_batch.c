for (size_t iter = 0; iter < (*lm)->params->n_iterations; ++iter)
{
	// sample subset of points (batch optimization)
	void* s_alloc = calloc((*lm)->params->batch_size, sizeof(size_t));
	if (!s_alloc)
		err("Couldn't allocate memory when trying to sample data.");
	size_t* sampled_idx = s_alloc;
	Matrix* X_sample = mat_sample(X, (*lm)->params->batch_size, false, sampled_idx);
	Matrix* Y_sample = NULL;
	mat_init(&Y_sample, (*lm)->params->batch_size, 1);
	for (size_t r = 0; r < (*lm)->params->batch_size; ++r)
		mat_set(&Y_sample, r, 0, mat_at(Y, sampled_idx[r], 0));

	free(sampled_idx);
	sampled_idx = NULL;

	Matrix* Yhat = NULL;
	mat_init(&Yhat, (*lm)->params->batch_size, 1);

	// get linear combination of data and weights
	mat_multiply_inplace(X_sample, (*lm)->W, &Yhat);
	
	// apply activation
	(*lm)->activation(&Yhat, *lm);

	// compute loss and check early stop criteria
	float loss = (*lm)->loss(Y_sample, Yhat, *lm);

	stop_early = __check_loss_tolerance(loss, previous_loss, (*lm)->params->early_stop_threshold, &tolerance_counter, (*lm)->params->early_stop_iterations);
	if (stop_early)
	{
		// incase early stop, we must release memory early
		mat_free(&X_sample);
		mat_free(&Y_sample);
		mat_free(&Yhat);
		break;
	}
	previous_loss = loss;

	// only print loss 10 times for any given number
	// of iterations
	if (iter % (size_t)((float)(*lm)->params->n_iterations / 10.0f) == 0)
	{
		if (iter == 0)
			initial_loss = loss;
		else if (iter > 0 && loss > 10 * initial_loss)
		{
			printf("WARNING: loss blew up. Consider lowering your learning rate.\n");
			// incase early stop, we must release memory early
			mat_free(&X_sample);
			mat_free(&Y_sample);
			mat_free(&Yhat);
			break;
		}
		if (verbose)
			printf("Loss at iteration %zu: %f\n", iter, loss);
	}	


	(*lm)->loss_gradient(Y_sample, Yhat, X_sample, *lm, &loss_grad);
	mat_free(&X_sample);
	mat_free(&Y_sample);
	mat_free(&Yhat);

	// update weights
	mat_multiply_s(&loss_grad, (*lm)->params->learning_rate);
	mat_subtract_e(&(*lm)->W, loss_grad);
}
