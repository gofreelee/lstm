add_executable(alexnet_test 
alexnet_test.cc
)
target_link_libraries(alexnet_test PUBLIC pthread cudart gtest gtest_main kernels models test_utils)
add_custom_command(TARGET alexnet_test
POST_BUILD
COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/data ${CMAKE_CURRENT_BINARY_DIR} 
VERBATIM
)
add_test(alexnet_test alexnet_test)