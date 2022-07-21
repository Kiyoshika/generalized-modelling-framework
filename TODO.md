* add absolute loss, hinge loss, huber loss
* introduce vectorized regression models for `f : R^w -> R^s` multidimensional output
	* softmax could be introduced for multiclassification in this case
* toggle verbosity when fitting models
* model regularization
* metrics like RMSE, MAE, confusion matrix, etc.
* add global install target
* rename final library output to `libgmf.a`

IN PROGRESS
* added regularization functions & gradients, need to add them into the models & losses
