include_directories(${CMAKE_CURRENT_LIST_DIR}/.)

cuda_add_library(LstmTestData
    LstmTestData.cpp)
add_subdirectory(wavefront_lstm_10_100_256_test)
add_subdirectory(wavefront_lstm_20_100_256_test)
# add_subdirectory(lstm_cell_test)
# add_subdirectory(lstm_cell_net_test)
