add_executable(lstm_cell_test lstm_cell_test.cc)
add_executable(lstm_cellf4_test lstm_cellf4_test.cc)

target_link_libraries(lstm_cell_test GTest::gtest_main LSTMCell IctLstmCell_cudalib LstmTestData)
target_link_libraries(lstm_cellf4_test GTest::gtest_main LSTMCellF4 IctLstmCell_cudalib LstmTestData)

add_custom_command(TARGET lstm_cellf4_test
POST_BUILD
COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/test_data ${CMAKE_CURRENT_BINARY_DIR} 
VERBATIM
)
