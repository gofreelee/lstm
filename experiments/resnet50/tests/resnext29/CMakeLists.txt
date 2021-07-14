add_executable(resnext29_test 
resnext29_test.cc
)
target_link_libraries(resnext29_test PUBLIC pthread cudart gtest gtest_main kernels models test_utils)
add_custom_command(TARGET resnext29_test
POST_BUILD
COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/data ${CMAKE_CURRENT_BINARY_DIR} 
VERBATIM
)
add_test(resnext29_test resnext29_test)