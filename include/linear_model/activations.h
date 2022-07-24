#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stddef.h>

/*
 * NOTE:
 * For all activations, we can assume we are given the
 * linear combination of X and W. So all parameters will
 * be XW that have shape (r, 1)
 */

// forward declaration
typedef struct Matrix Matrix;
typedef struct LinearModel LinearModel;

// f(x) = x
void gmf_activation_identity(
		Matrix** XW, 
		const LinearModel* lm);

// f(x) = 1 / (1 + e^(-x))
// NOTE: this produces the raw sigmoid output with no cutoff.
// If you want a cutoff, use gmf_activation_sigmoid_hard()
void gmf_activation_sigmoid_soft(
		Matrix** XW,
		const LinearModel* lm);

// f(x) = 1 / (1 + e^(-x))
// NOTE: this forces the output into a hard 0-1 based on a cutoff.
// The cutoff is determined by sigmoid_threshold param in LinearModel 
// If you don't want the cutoff, use gmf_activation_sigmoid_soft()
void gmf_activation_sigmoid_hard(
		Matrix** XW,
		const LinearModel* lm);

#endif
