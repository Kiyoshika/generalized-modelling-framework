# UTIL
add_library(gmf_util gmf_util.c)
target_include_directories(util PUBLIC ${GMF_SOURCE_DIR}/include)
target_include_directories(util PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(gmf_util matrix)

# METRICS
add_library(metrics metrics.c)
target_include_directories(metrics PUBLIC ${GMF_SOURCE_DIR}/include)
target_include_directories(metrics PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(metrics matrix)

# LINEAR MODEL
add_library(linear_model 
	linear_model/linear_model.c
	linear_model/activations.c
	linear_model/losses.c
	linear_model/loss_gradients.c
	linear_model/regularization.c
	linear_model/regularization_gradient.c)
target_include_directories(linear_model PUBLIC ${GMF_SOURCE_DIR}/include/linear_model)
target_include_directories(linear_model PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(linear_model m gmf_util matrix)

# LINEAR MODEL (OVR)
add_library(linear_model_ovr
	linear_model/linear_model_ovr.c)
target_include_directories(linear_model_ovr PUBLIC ${GMF_SOURCE_DIR}/include/linear_model)
target_include_directories(linear_model_ovr PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(linear_model_ovr linear_model gmf_util matrix)

# DISTANCES
add_library(distance neighbors/distances.c)
target_include_directories(distance PUBLIC ${GMF_SOURCE_DIR}/include/neighbors)
target_include_directories(distance PUBLIC ${CMatrix_SOURCE_DIR}/include/vector)
target_link_libraries(distance vector m)

# KNN
add_library(knn neighbors/knn.c)
target_include_directories(knn PUBLIC ${GMF_SOURCE_DIR}/include/neighbors)
target_include_directories(knn PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_include_directories(knn PUBLIC ${CMatrix_SOURCE_DIR}/include/vector)
target_link_libraries(knn distance vector matrix)

# FULL LIBRARY
add_library(gmf gmf.c)
target_link_libraries(gmf
	linear_model
	linear_model_ovr
	gmf_util
	matrix)

# examples for linear model
if (${EXAMPLES})
	add_executable(least_squares linear_model/examples/least_squares.c)
	target_link_libraries(least_squares linear_model metrics)
	set_target_properties(least_squares PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples/linear_model")

	add_executable(regularization linear_model/examples/regularization.c)
	target_link_libraries(regularization linear_model metrics)
	set_target_properties(regularization PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples/linear_model")

	add_executable(logistic_regression linear_model/examples/logistic_regression.c)
	target_link_libraries(logistic_regression linear_model metrics)
	set_target_properties(logistic_regression PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples/linear_model")

	add_executable(multiclass_classification linear_model/examples/multiclass_classification.c)
	target_link_libraries(multiclass_classification linear_model_ovr metrics)
	set_target_properties(multiclass_classification PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples/linear_model")

	add_executable(classic_knn neighbors/examples/classic_knn.c)
	target_link_libraries(classic_knn knn)
	set_target_properties(classic_knn PROPERTIES RUNTIME_OUTPUT_DIRECTORY "examples/neighbors")
endif()
