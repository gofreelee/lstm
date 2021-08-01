add_executable(TestSeq2Seq TestSeq2Seq.cc)
add_executable(BenchmarkSeq2Seq BenchmarkSeq2Seq.cc)
target_link_libraries(TestSeq2Seq GTest::gtest_main Seq2SeqNet LSTMCell IctSeq2SeqLstm_cudalib LstmTestData)
target_link_libraries(BenchmarkSeq2Seq GTest::gtest_main Seq2SeqNet LSTMCell IctSeq2SeqLstm_cudalib LstmTestData)

add_custom_command(TARGET BenchmarkSeq2Seq
POST_BUILD
COMMAND cp -r ${CMAKE_CURRENT_LIST_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/.
VERBATIM
)
add_test(TestSeq2Seq TestSeq2Seq)
