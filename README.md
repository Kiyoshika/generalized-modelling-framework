# Generalized Modelling Framework (libgmf)
This is a revival of an old project. I felt like my old API was a bit messy and also wanted to build this library on top of my linear algebra library to make computations a bit easier.

The goal of this library is to make a very modular modelling framework for building predictive models by swapping out bits and pieces (activation functions, loss functions, etc.) as well as making it very easy to define custom user activations, losses and such.

# Building from Source
This library depends on my other library, CMatrix. To properly clone, use

`git clone --recursive git@github.com:Kiyoshika/generalized-modelling-framework.git`

Then follow the typical CMake prodecure:

* `cd generalized-modelling-framework`
* `mkdir build && cd build`
* `cmake ..`
* `make`

This will generate a `libgmf.a` static library. At the moment I have no global install targets but will add later.
