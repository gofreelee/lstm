add_executable(resnet50_test 
resnet50_test.cc
)
target_link_libraries(resnet50_test PUBLIC pthread cudart gtest gtest_main kernels models test_utils)
add_custom_command(TARGET resnet50_test
POST_BUILD
COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/data ${CMAKE_CURRENT_BINARY_DIR} 
VERBATIM
)
add_test(resnet50_test resnet50_test)