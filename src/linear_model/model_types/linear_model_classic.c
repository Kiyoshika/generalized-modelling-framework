for (size_t iter = 0; iter < (*lm)->params->n_iterations; ++iter)
{
	Matrix* Yhat = NULL;
	mat_init(&Yhat, X->n_rows, 1);

	// get linear combination of data and weights
	mat_multiply_inplace(X, (*lm)->W, &Yhat);
	
	// apply activation
	(*lm)->activation(&Yhat);

	// compute loss and check early stop criteria
	float loss = (*lm)->loss(Y, Yhat, *lm);
	stop_early = __check_loss_tolerance(loss, previous_loss, (*lm)->params->early_stop_threshold, &tolerance_counter, (*lm)->params->early_stop_iterations);
	if (stop_early)
	{
		// incase of early stop, free memory early
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
			// incase of early stop, free memory early
			printf("WARNING: loss blew up. Consider lowering your learning rate.\n");
			mat_free(&Yhat);
			break;
		}

		if (verbose)
			printf("Loss at iteration %zu: %f\n", iter, loss);
	}	


	(*lm)->loss_gradient(Y, Yhat, X, *lm, &loss_grad);
	mat_free(&Yhat);

	// update weights
	mat_multiply_s(&loss_grad, (*lm)->params->learning_rate);
	mat_subtract_e(&(*lm)->W, loss_grad);
}
