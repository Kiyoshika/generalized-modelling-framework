for (size_t iter = 0; iter < (*lm)->params->n_iterations; ++iter)
{
	// sample single point (stochastic optimization)
	size_t sampled_idx[1];
	Matrix* X_sample = mat_sample((*lm)->X, 1, false, sampled_idx);
	Matrix* Y_sample = mat_subset(Y, sampled_idx[0], sampled_idx[0], 0, 0);

	Matrix* Yhat = NULL;
	mat_init(&Yhat, 1, 1);

	// get linear combination of data and weights
	mat_multiply_inplace(X_sample, (*lm)->W, &Yhat);
	
	// apply activation
	(*lm)->activation(&Yhat);

	// compute loss and check early stop criteria
	float loss = (*lm)->loss(Y_sample, Yhat);

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

		printf("Loss at iteration %zu: %f\n", iter, loss);
	}	


	(*lm)->loss_gradient(Y_sample, Yhat, X_sample, &loss_grad);
	mat_free(&X_sample);
	mat_free(&Y_sample);
	mat_free(&Yhat);

	// update weights
	mat_multiply_s(&loss_grad, (*lm)->params->learning_rate);
	mat_subtract_e(&(*lm)->W, loss_grad);
}
