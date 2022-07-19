# Generalized Modelling Framework (libgmf)
This is a revival of an old project. I felt like my old API was a bit messy and also wanted to build this library on top of my linear algebra library to make computations a bit easier.

The goal of this library is to make a very modular modelling framework for building predictive models by swapping out bits and pieces (activation functions, loss functions, etc.) as well as making it very easy to define custom user activations, losses and such.

# Building from Source
This library depends on my other library, CMatrix. To properly clone, use

`git clone --recursive git@github.com:Kiyoshika/generalized-modelling-framework.git`

Then follow the typical CMake prodecure:

* `cd generalized-modelling-framework`
* `mkdir build && cd build`

If you want to compile all examples, pass the `-DEXAMPLES=ON` to CMake, otherwise ignore

* `cmake ..` or `cmake -DEXAMPLES=ON ..`
* `make`

This will generate a `libgmf.a` static library. At the moment I have no global install targets but will add later.

# Documentation (In Progress)
This is in-progress documentation as I develop the library.

## Namespace Structure
Since C doesn't have namespaces, functions are organized into "namespaces" with underscore prefixes.

All functions of the library start with `gmf` (generalized modelling framework). Afterwards, the structure is along the lines of:

`gmf_[namespace1]_[namespace2]_[funcs]` - NOTE: some functions don't have a `namespace2`

where `namespace1` and `namespace2` are:
* `model` - contains all models
	* `linear` - linear model
	* `linear_ovr` - one-vs-rest linear model for multi classification
	* `linear_vec` - vectorized linear models for multi-dimensional output (coming soon)
* `util` - contains utility functions
* `metrics` - contains metrics to evaluate models (coming soon)
* `activation` - contains all activation functions used in linear models
* `loss` - contains all loss functions used in linear models
* `loss_gradient` - contains all loss function gradients used in linear models

and `funcs` represents all function names inside those two namespaces. See header files for exact names.

## Linear Models
Linear models (at the moment) are separated into two categories:
* classic model
* OVR model
* vector model (coming soon - undocumented)

The classic model is the traditional linear model, `f : R^w -> R` where `w` is the # of features. This model supports classification and regression problems.

The OVR (one-vs-rest) model is a wrapper around the classic model defined as `f : R^w -> Z*`. Note the domain is now `Z*` which is the set of non-negative integers. This model ONLY supports classification problems, specifically ones with multiple classes in the set of non-negative integers `{0, 1, 2, ...}`

OVR introduces a couple additional members:
* `n_models` - the total number of submodels equal to `c choose 2` where `c` is the number of classes
* `models` - an array of pointers to `LinearModel` accessed by `ovr_model->models[i]->...`

There are other members but not necessarily useful to the user. See [linear_model_ovr.h](include/linear_model/linear_model_ovr.h) for more details.

### Parameters
Linear models support a set of parameters defined below with their default values:
* `n_iterations: 1000` - # of iterations while training model
* `learning_rate: 0.001f` - softening parameter for the loss gradient when updating weights
* `early_stop_threshold: 0.001f` - maximum threshold for the difference between losses each iteration to determine an early stop (see below)
* `early_stop_iterations: n_iterations / 10` - minimum number of consecutive iterations where the difference in loss is below `early_stop_threshold`. Once this is reached, the model stops training early as it appears to have converged. If this is not met and `n_iterations` is complete, a warning as printed notifying the user that the model may not have converged yet.
* `model_type: CLASSIC` - one of `CLASSIC`, `BATCH` or `STOCHASTIC` determining how to optimize the model. `CLASSIC` uses the entire training data each iteration, `BATCH` uses `batch_size` random data points per iteration and `STOCHASTIC` uses a single random data point per iteration.
* `batch_size: n_rows / 4` - number of random data points to sample each iteration. Only used if `BATCH` is selected for `model_type`. `n_rows` represents the number of rows in the training set.

For classic models, parameters can be set as follows:
```c
LinearModel* lm = ...
lm->params->[param_name] = [value];
```

For OVR models, you can either set a parameter individually per model or use the built-in setters to apply a parameter to all models
```c
LinearModelOVR* lm = ...

// individually
for (size_t m = 0; m < lm->n_models; ++m)
	lm->models[m]->params->[param_name] = [value];

// using built-in setters
gmf_model_linear_ovr_set_[param_name](&lm, [value]);
```

### Examples
For examples on usage for linear model, see [Linear Model Examples](src/linear_model/examples)
