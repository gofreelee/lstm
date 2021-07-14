#pragma once

#include "cuda_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "[%s:%d] Assertion failed:", __FILE__, __LINE__);  \
            fputs(#x "\n", stderr);                                            \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define CHECK_EQ(val1, val2, message)                                          \
    do {                                                                       \
        if (val1 != val2) {                                                    \
            std::cerr << __FILE__ << "(" << __LINE__ << "): " << message       \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(condition)                                                  \
    do {                                                                       \
        cudaError_t error = condition;                                         \
        CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error));               \
    } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
