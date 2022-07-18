* add absolute loss, hinge loss, huber loss
* introduce multi-class support with sigmoid & cross entropy
* toggle verbosity when fitting models
* model regularization
* add global install target
* rename final library output to `libgmf.a`

# IN PROGRESS - OVR models (one vs rest)
Got basic allocator working. Need to:
* implement filtering matrix for filtering classes
	* maybe add this as a feature in CMatrix?
* translate class pairs to 0,1 when evaluating activation & loss and back when making predictions
* add setters for different parameters to apply those parameters to all models
	* e.g., `gmf_model_linear_ovr_set_activation(ovr_model, &gmf_activation_sigmoid)` 
