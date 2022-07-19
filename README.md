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

**CONTENTS**
* [Namespace Structure](#namespace-structure)
* [Linear Models](#linear-models)
	* [Memory Management](#memory-management)
	* [Activation Functions](#activation-functions)
	* [Loss Functions](#loss-functions)
	* [Parameters](#parameters)
	* [Examples](#examples)

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

Linear models are optimized via gradient descent. All models MUST have an activation, loss and loss gradient set. See below sections.

The classic model is the traditional linear model, `f : R^w -> R` where `w` is the # of features. This model supports classification and regression problems.

The OVR (one-vs-rest) model is a wrapper around the classic model defined as `f : R^w -> Z*`. Note the domain is now `Z*` which is the set of non-negative integers. This model ONLY supports classification problems, specifically ones with multiple classes in the set of non-negative integers `{0, 1, 2, ...}`

OVR introduces a couple additional members:
* `n_models` - the total number of submodels equal to `c choose 2` where `c` is the number of classes
* `models` - an array of pointers to `LinearModel` accessed by `ovr_model->models[i]->...`

There are other members but not necessarily useful to the user. See [linear_model_ovr.h](include/linear_model/linear_model_ovr.h) for more details.

### Memory Management
All model initializers allocate memory internally and return a pointer. You can use it as follows:
```c
LinearModel* lm = gmf_model_linear_init();
LinearModelOVR* ovr = gmf_model_linear_ovr_init();
```

You must free the memory using the respective free functions:
```c
gmf_model_linear_free(&lm);
gmf_model_linear_ovr_free(&ovr);
```

### Activation Functions
These are the current supported activation functions. You can set an activation function as follows:
```c
/* classic model */
LinearModel* lm = gmf_model_linear_init();
lm->activation = &gmf_activation_...

/* OVR model */
LinearModelOVR* lm = gmf_model_linear_ovr_init();

// OPTION 1 - manually set for an individual model
lm->models[i]->activation = &gmf_activation_... // manually set for an individual submodel

// OPTION 2 - use built-in setters to apply to ALL submodels
gmf_model_linear_ovr_set_activation(&lm, &gmf_activation_...); // set for ALL submodels
```

* `gmf_activation_identity` - identity activation typically used in continuous regression, `f(x) = x`
* `gmf_activation_sigmoid` - constrains output to [0, 1] typically used in classification, `f(x) = 1 / (1 + e^(-x))`

### Loss Functions
These are the current supported loss functions (and their gradients). You can set a loss function (+ gradient) function as follows:
```c
// classic model
LinearModel* lm = ...
lm->loss = &gmf_loss_...
lm->loss_gradient = &gmf_loss_gradient_...

// OVR model
LinearModelOVR* lm = ...

// OPTION 1 - manually set for an individual model
lm->models[i]->loss = &gmf_loss_...
lm->models[i]->loss = &gmf_loss_gradient_...

// OPTION 2 - use built-in setters to apply to ALL submodels
gmf_model_linear_ovr_set_loss(&lm, &gmf_loss_...);
gmf_model_linear_ovr_set_loss_gradient(&lm, &gmf_loss_gradient_...);
```

* `gmf_loss_squared` - squared loss, typically used in regression problems, `L(y, yhat) = (y - yhat)^2`
* 'gmf_loss_cross_entropy` - typically used with continuous values in [0, 1], `L(y, yhat) = -ylog(yhat) - (1 - y)log(1 - yhat)`

The gradients follow the same naming convention as above, except use `loss_gradient` instead of just `loss_`

### Parameters
Linear models support a set of parameters defined below with their default values:
* `n_iterations: 1000` - # of iterations while training model
* `learning_rate: 0.001f` - softening parameter for the loss gradient when updating weights
* `early_stop_threshold: 0.001f` - maximum threshold for the difference between losses each iteration to determine an early stop (see below)
* `early_stop_iterations: n_iterations / 10` - minimum number of consecutive iterations where the difference in loss is below `early_stop_threshold`. Once this is reached, the model stops training early as it appears to have converged. If this is not met and `n_iterations` is complete, a warning as printed notifying the user that the model may not have converged yet. NOTE: if you want to disable early stop, you can set it equal to `n_iterations`.
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
