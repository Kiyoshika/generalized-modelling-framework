#include "distances.h"
#include "vector.h"

void __check_valid_vecs(const Vector* x, const Vector* y)
{
	if (x->n_elem != y->n_elem)
	{
		printf("Cannot measure distance of vectors with different sizes.");
		exit(-1);
	}
}

static float __euclidean(float x, float y)
{
	return (x - y) * (x - y);
}

static float __manhattan(float x, float y)
{
	return fabsf(x - y);
}

static float __compute_distance(
		const Vector* x,
		const Vector* y,
		float (*distance)(float, float))
{
	__check_valid_vecs(x, y);

	float sum = 0.0f;
	for (size_t i = 0; i < x->n_elem; ++i)
	{
		float xi = vec_at(x, i);
		float yi = vec_at(y, i);
		sum += distance(xi, yi);
	}

	return sum;
}

float gmf_distance_euclidean(const Vector* x, const Vector* y)
{
	return sqrt(__compute_distance(x, y, &__euclidean));
}

float gmf_distance_manhattan(const Vector* x, const Vector* y)
{
	return __compute_distance(x, y, &__manhattan);	
}
