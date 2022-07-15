add_library(linear_model 
	linear_model/linear_model.c
	linear_model/activations.c
	linear_model/losses.c
	linear_model/loss_gradients.c)
target_include_directories(linear_model PUBLIC ${GMF_SOURCE_DIR}/include/linear_model)
target_include_directories(linear_model PUBLIC ${CMatrix_SOURCE_DIR}/include/matrix)
target_link_libraries(linear_model matrix)

add_executable(testing testing.c)
target_link_libraries(testing linear_model)
