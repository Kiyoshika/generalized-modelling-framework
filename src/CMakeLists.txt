add_library(linear_model 
	linear_model/linear_model.c
	linear_model/activations.c
	linear_model/losses.c
	linear_model/loss_gradients.c)
target_include_directories(linear_model PUBLIC ${GMF_SOURCE_DIR}/include/linear_model)
target_include_directories(linear_model PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(linear_model matrix)

# examples for linear model
add_executable(linear_model_basic examples/linear_model_basic.c)
target_link_libraries(linear_model_basic linear_model)

