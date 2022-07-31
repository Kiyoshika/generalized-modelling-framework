// for every test data point, compare distances with every
// training point
for (size_t tr = 0; tr < knn->X->n_rows; ++tr)
{
	// don't compute distance to self
	if (tr == r)
		continue;

	test_row = mat_get_row(knn->X, tr);
	float distance = knn->params->distance(row_vec, test_row);
	distance_pair dp = {.distance = distance, .idx = tr};
	distance_pairs[tr] = dp;
	vec_free(&test_row);
}
