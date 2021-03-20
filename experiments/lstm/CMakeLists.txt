add_executable(lstm_cuda_test lstm_cuda_test)
target_link_libraries(lstm_cuda_test absl::status GTest::gtest_main)
add_test(lstm_cuda_test lstm_cuda_test)