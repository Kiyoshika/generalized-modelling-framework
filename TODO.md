* add absolute loss, hinge loss, huber loss
* introduce vectorized regression models for `f : R^w -> R^s` multidimensional output
	* softmax could be introduced for multiclassification in this case
* rework linear models to not store a copy of X and instead rely on user using `gmf_util_add_bias`
	* this will make it especially more efficient for OVR models
	* can also check the first column of a data set is all 1's when fitting to warn user if a bias term isn't detected.
* toggle verbosity when fitting models
* model regularization
* metrics like RMSE, MAE, confusion matrix, etc.
* add global install target
* rename final library output to `libgmf.a`
