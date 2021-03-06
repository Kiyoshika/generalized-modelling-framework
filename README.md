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

This will generate a libgmf.a static library you can link to other projects.

# Including in Other Projects
This will guide you on how to include this library directly in your other CMake projects if you don't want to compile separately and link.

This assumes you have a very barebones CMake setup with a `CMakeLists.txt`

1. Create a folder to store external dependencies (e.g., `ext`) and use `git submodule add https://github.com/Kiyoshika/generalized-modelling-framework.git`
2. To fetch dependencies, use `git submodule update --init --recursive` - this will fetch the dependent submodules GMF uses
3. In your root CMakeLists.txt you can do the following to link the library & headers to your target
```cmake
# add GMF to your build
add_subdirectory(ext/generalized-modelling-framework)

# this now assumes you have a target "mytarget"
target_link_libraries(mytarget gmf)
target_include_directories(mytarget PUBLIC ${GMF_SOURCE_DIR}/include)
```

Afterwards, you should be able to use headers such as `#include "linear_models.h"` for example.

# Documentation (In Progress)
This is in-progress documentation as I develop the library.

**CONTENTS**
* [Namespace Structure](#namespace-structure)
* [Metrics](#metrics)
* [Linear Models](#linear-models)
 	* [Bias Term](#bias-term)
	* [Memory Management](#memory-management)
	* [Activation Functions](#activation-functions)
	* [Loss Functions](#loss-functions)
	* [Parameters](#parameters)
	* [Class Weights](#class-weights)
	* [Regularization](#regularization)
	* [Examples](#examples)
* [Neighbor Models](#neighbor-models)
	* [Nearest Neighbors](#nearest-neighbors)
	* [Examples](#examples-neighbors)

## Namespace Structure
Since C doesn't have namespaces, functions are organized into "namespaces" with underscore prefixes.

All functions of the library start with `gmf` (generalized modelling framework). Afterwards, the structure is along the lines of:

`gmf_[namespace1]_[namespace2]_[funcs]` - NOTE: some functions don't have a `namespace2`

where `namespace1` and `namespace2` are:
* `model` - contains all models
	* `linear` - linear model
	* `linear_ovr` - one-vs-rest linear model for multi classification
	* `linear_vec` - vectorized linear models for multi-dimensional output (coming soon)
	* `knn` - K nearest neighbor model
* `util` - contains utility functions
* `metrics` - contains metrics to evaluate models
* `activation` - contains all activation functions used in linear models
* `loss` - contains all loss functions used in linear models
* `loss_gradient` - contains all loss function gradients used in linear models
* `regularization` - contains all regularization functions used in linear models
* `regularization_gradient` - contains all regularization gradients used in linear models
* `distance` - contains all distance functions used in neighbor-based models

and `funcs` represents all function names inside those two namespaces. See header files for exact names.

## Metrics
There are a few built-in metrics for classification and regression tasks.

Each metrics function takes two `Matrix*` arguments for the actuals and predicted. There is an additional `void*` parameter for any additional parameters that would need to be passed for the metric to use. Currently only the confusion matrix requires this to determine the # of classes.

Below are the current built-in metrics:

* `gmf_metrics_mae` - mean absolute error, f(y, yhat) = mean(|y - yhat|)
	* requires no additional parameters
* `gmf_metrics_mse` - mean squared error, f(y, yhat) = mean((y - yhat)^2)
	* requires no additional parameters
* `gmf_metrics_confusion_matrix` - prints a confusion matrix to the console and returns weighted f1 score.
	* requires a `size_t*` parameter specifying the # of classes 

See [Linear Model Examples](src/linear_model/examples) for usage of mean absolute error and confusion matrix.

## Linear Models
HEADER: `#include "linear_models.h"`

Linear models (at the moment) are separated into two categories:
* [classic model](include/linear_model/linear_model.h)
* [OVR model](include/linear_model/linear_model_ovr.h)
* vector model (coming soon - undocumented)

Linear models are optimized via gradient descent. All models MUST have an activation, loss and loss gradient set. See below sections.

The classic model is the traditional linear model, `f : R^w -> R` where `w` is the # of features. This model supports classification and regression problems.

The OVR (one-vs-rest) model is a wrapper around the classic model defined as `f : R^w -> Z*`. Note the domain is now `Z*` which is the set of non-negative integers. This model ONLY supports classification problems, specifically ones with multiple classes in the set of non-negative integers `{0, 1, 2, ...}`

OVR introduces a couple additional members:
* `n_models` - the total number of submodels equal to `c choose 2` where `c` is the number of classes
* `models` - an array of pointers to `LinearModel` accessed by `ovr_model->models[i]->...`

OVR also supports adjusting class weights as it's a classification-focused optimizer. See [Class Weights](#class-weights) for more.

There are other members but not necessarily useful to the user. See [linear_model_ovr.h](include/linear_model/linear_model_ovr.h) for more details.

### Bias Term
It's recommended for your input data to have a bias term, otherwise the model will be forced to pass through the origin `(0, 0, ..., 0)`.

To add a bias term, you can use `gmf_util_add_bias(&X)` where `X` is a `Matrix*`. See examples for specific usage of this.

Note that this will invalidate the previous pointer to `X`, so be cautious if you have any other pointers to `X` prior to adding the bias term.

### Memory Management
All model initializers allocate memory internally and return a pointer. You can use it as follows:
```c
LinearModel* lm = gmf_model_linear_init();

// OVR requires to specify # of classes and class weights upfront
// using NULL for class weights will compute them automatically.
// see section on class weights for more detail
LinearModelOVR* ovr = gmf_model_linear_ovr_init(5, NULL);
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
LinearModel* lm = ...
gmf_model_linear_set_activation(&lm, &gmf_activation_...);

/* OVR model */
LinearModelOVR* lm = ...
gmf_model_linear_ovr_set_activation(&lm, &gmf_activation_...);
```

* `gmf_activation_identity` - identity activation typically used in continuous regression: `f(x) = x`
* `gmf_activation_sigmoid_soft` - constrains output to [0, 1] continuously; typically used in classification: `f(x) = 1 / (1 + e^(-x))`
* `gmf_activation_sigmoid_hard` - constrains output to a hard 0 or 1 discretely; used in classification: `f(x) = 1 / (1 + e^(-x))`
	* the cutoff threshold can be controlled with `sigmoid_threshold`. See parameter section for more detail

### Loss Functions
These are the current supported loss functions (and their gradients). You can set a loss function (+ gradient) function as follows:
```c
/* classic model */
LinearModel* lm = ...
gmf_model_linear_set_loss(&lm, &gmf_loss_...);
gmf_model_linear_set_loss_gradient(&lm, &gmf_loss_gradient_...);

/* OVR model */
LinearModelOVR* lm = ...
gmf_model_linear_ovr_set_loss(&lm, &gmf_loss_...);
gmf_model_linear_ovr_set_loss_gradient(&lm, &gmf_loss_gradient_...);
```

* `gmf_loss_squared` - typically used in regression problems, `L(y, yhat) = (y - yhat)^2`
* `gmf_loss_absolute` - typically used in regression problems, `L(y, yhat) = |y - yhat|`
* `gmf_loss_cross_entropy` - typically used with continuous values in [0, 1], `L(y, yhat) = -ylog(yhat) - (1 - y)log(1 - yhat)`
* `gmf_loss_hinge` - used in classification problems. NOTE: this loss expects a hard 0 or 1 output. Formula taken from [Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss) 
* `gmf_loss_huber` - used in classification or regression problems. Formula taken from [Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)
	* This takes a `huber_delta` parameter, see parameters section for more detail.
	* NOTE: for classification, people typically use a "modified huber" loss which is not implemented (yet) 

The gradients follow the same naming convention as above, except use `loss_gradient` instead of just `loss_`

### Parameters
Linear models support a set of parameters defined below with their default values:
* `n_iterations: 1000` - # of iterations while training model
* `learning_rate: 0.001f` - softening parameter for the loss gradient when updating weights
* `early_stop_threshold: 0.001f` - maximum threshold for the difference between losses each iteration to determine an early stop (see below)
* `early_stop_iterations: n_iterations / 10` - minimum number of consecutive iterations where the difference in loss is below `early_stop_threshold`. Once this is reached, the model stops training early as it appears to have converged. If this is not met and `n_iterations` is complete, a warning as printed notifying the user that the model may not have converged yet. NOTE: if you want to disable early stop, you can set it equal to `n_iterations`.
* `model_type: CLASSIC` - one of `CLASSIC`, `BATCH` or `STOCHASTIC` determining how to optimize the model. `CLASSIC` uses the entire training data each iteration, `BATCH` uses `batch_size` random data points per iteration and `STOCHASTIC` uses a single random data point per iteration.
* `batch_size: n_rows / 4` - number of random data points to sample each iteration. Only used if `BATCH` is selected for `model_type`. `n_rows` represents the number of rows in the training set.
* `huber_delta: 1.0` - A hyperparameter for `gmf_loss_huber`
* `sigmoid_threshold: 0.5` - A hyperparameter for `gmf_activation_sigmoid_hard`. If the output of sigmoid is above this threshold, the label is converted to 1 and 0 otherwise.

Parameters can be set as follows - they typically follow the same naming convention as the above names.
```c
/* classic model */
LinearModel* lm = ...
gmf_model_linear_set_iterations(&lm, 1000);
gmf_model_linear_set_learning_rate(&lm, 0.005f);

/* OVR model */
LinearModelOVR* lm = ...
gmf_model_linear_ovr_set_iterations(&lm, 1000);
gmf_model_linear_ovr_set_learning_rate(&lm, 0.005f);
```

You can also toggle verbosity while training. If verbosity is true, it will display the loss function throughout training. You can toggle this in the third parameter in the fit functions:

```c
gmf_model_linear_fit(X, Y, true); // or false
gmf_model_linear_ovr_fit(X, Y, true); // or false
```

If verbosity is off, it will still print warnings if the loss function hasn't "converged" and a note if the convergence stopped early.

If desired, with OVR models you can change the individual model parameters instead of blanketing all models with the same parameter.
```c
// update the i-th model's parameter in an OVR model
gmf_model_linear_set_[param](&ovr_model->models[i], [value]);
```

### Class Weights
OVR models support adjusting class weights. You can either manually specify weights or have them calculated automatically.

**WARNING**: You may notice the `LinearModel` struct contains class weights but it is NOT DESIGNED to be used by itself. Class weights areNOT free'd in `LinearModel` and will lead to a memory leak. This is because, as mentioned, OVR models are a wrapper around `LinearModel` and require a pointer to that data - everything is free'd correctly in `gmf_model_linear_ovr_free`.

The automatic calculation for class weights is `N / (n_classes * class_size)` where
* `N` is the total number of data points
* `n_classes` is the total number of classes
* `class_size` is the total number of a particular class (e.g., 0, 1, 2, etc.)

```c
// create OVR model with 3 classes {0, 1, 2}
// passing NULL to class weights will compute them automatically
// otherwise you can pass an array of floats to control class weights
LinearModelOVR* ovr_model = gmf_model_linear_ovr_init(3, NULL);

// to force class weights to be uniform, you can make them the same value
// note that class weights do not need to add up to one
float class_weights[3] = {1.0f, 1.0f, 1.0f};
LinearModelOVR* uniform_weights = gmf_model_linear_ovr_init(3, class_weights);
```

### Regularization
Regularization adds an additional penalty term to the loss function as an attempt to aid overfitting. By default, the library supports L1 to LN regularization, but users can define their own functions if desired.

If using regularization, you must specify:
* Regularization function `gmf_regularization_...`
* Regularization gradient `gmf_regularization_gradient_...`
* Regularization params `float*`

The `params` are values passed to the regularization functions. In the case of L1 & L2 this is just the lambda parameter, so it's a `float[1]` array you'd pass.

For a full example, see [Regularization Example](src/linear_model/examples/regularization.c).

### Examples
For examples on usage for linear model, see [Linear Model Examples](src/linear_model/examples)

## Neighbor Models
This is a collection of models that are based on neighboring data points.

### Nearest Neighbors
HEADER: [#include "knn.h"](include/neighbors/knn.h) 

```c
// create new KNN model
KNN* knn = gmf_model_knn_init();

// cleanup memory
gmf_model_knn_free(&knn);
```

Find the K closest data points and take the average of their target.

Data points are compared by [distance](include/neighbors/distances.h) functions (`gmf_distance_...`)

There are two types of KNN models:
* `CLASSIC` - the naive implementation where each test point has its distance compared to every training point
* `KDTree` - training data is stored in a KDTree structure to make traversing nearest neighbors quicker

You can set types with:
```c
KNN* knn = ...
gmf_model_knn_set_type(&knn, CLASSIC); // or KDTree
```

`CLASSIC` is the default type (for now, until KDTree is implemented)

#### Parameters
Below is a list of parameters and their defualt values for nearest neighbor model.

* `distance: euclidean` - distance function to compare two points. see [distance](include/neighbors/distances.h) functions for complete list
* `n_neighbors: 3` - number of neighbors to compare test points with training points

You can set parameters with:
```c
KNN* knn = ...
gmf_model_knn_set_[param](&knn, [value]);
```

Where `[param]` is one of
* `type` (see previous section)
* `distance`
* `neighbors` 

## Examples (Neighbors)
See the [examples](src/neighbors/examples) for neighbor models
