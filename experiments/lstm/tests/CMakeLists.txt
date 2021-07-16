include_directories(${CMAKE_CURRENT_LIST_DIR}/.)

cuda_add_library(LstmTestData
    LstmTestData.cpp)
add_subdirectory(naive_lstm_test)
add_subdirectory(onekernel_lstm_test)
add_subdirectory(wavefront_lstm_8_8_1_256_test)
add_subdirectory(wavefront_lstm_10_100_1_256_test)