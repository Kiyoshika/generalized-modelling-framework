#ifndef DISTANCES_H
#define DISTANCES_H

// forward declaration
typedef struct Vector Vector;

// D(x, y) = sqrt(sum_i (x_i - y_i)^2)
float gmf_distance_euclidean(const Vector* x, const Vector* y);

// D(x, y) = sum_i |x_i - y_i|
float gmf_distance_manhattan(const Vector* x, const Vector* y);
#endif
