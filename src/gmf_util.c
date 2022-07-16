#include "matrix.h"
#include "gmf_util.h"

void gmf_util_add_bias(Matrix** X)
{
	Matrix* X_copy = mat_copy(*X);
	mat_reshape(X, (*X)->n_rows, (*X)->n_columns + 1);
	for (size_t r = 0; r < (*X)->n_rows; ++r)
	{
		mat_set(X, r, 0, 1.0f);
		for (size_t c = 1; c < (*X)->n_columns; ++c)
			mat_set(X, r, c, mat_at(X_copy, r, c - 1));
	}	
	mat_free(&X_copy);
}
