#pragma once
#include "cuda_ops.h"

template <int kBatchSize, int kSize, bool kFuseRelu> class BottleneckAdd {
  public:
    typedef Tensor<kSize> StateType;
    static void Forward(const Tensor<kSize> *input1, Tensor<kSize> *input2,
                        StateType *state) {
        int grid_size = (kSize + kBlockSize - 1) / kBlockSize;
        void *args[] = {&input1, &input2, &state};
        cudaLaunchKernel((const void *)operator_vecaddvec_h<kSize, kFuseRelu>,
                         dim3(grid_size), dim3(kBlockSize), (void **)args, 0);
    }
};
