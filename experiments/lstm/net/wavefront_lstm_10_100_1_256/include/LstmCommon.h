#ifndef _LSTM_COMMON_H_
#define _LSTM_COMMON_H_

#include <cstdio>

#define cudaCheckErrors(msg)                                                   \
    do {                                                                       \
        cudaError_t __err = cudaGetLastError();                                \
        if (__err != cudaSuccess) {                                            \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,            \
                    cudaGetErrorString(__err), __FILE__, __LINE__);            \
            fprintf(stderr, "*** FAILED - ABORTING\n");                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

//#define CUDATEST_HOME             "/home/wqf/cuda-dir/cuda_test"
#define LSTM_RAMMER_CONSTANT_PATH                                              \
    CUDATEST_HOME "/experiment/lstm/lstm_rammer_constant/"
#define LSTM_RAMMER_BS4_CONSTANT_PATH                                          \
    CUDATEST_HOME "/experiment/lstm_batch_size_4/lstm_rammer_bs4_constant/"

#endif