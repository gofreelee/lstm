#pragma once
#include "cuda_ops.h"
#include "utils.h"

template <int kBatchSize, int kChannels, int kHeight, int kWidth, int kKernelH,
          int kKernelW, int kPadH, int kPadW, int kStrideH, int kStrideW>
class MaxPool {
  public:
    enum {
        kInputSize = kBatchSize * kChannels * kHeight * kWidth,
        kPoolHeight = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1,
        kPoolWidth = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1,
        kSize = kBatchSize * kChannels * kPoolHeight * kPoolWidth,
    };

    typedef MaxPoolState<kSize> StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        constexpr int grid_size = (kSize + kBlockSize - 1) / kBlockSize;
        void *args[] = {&input, &state};
        cudaLaunchKernel(
            (const void *)operator_max_pool_h<
                kBatchSize, kChannels, kHeight, kWidth, kKernelH, kKernelW,
                kPadH, kPadW, kStrideH, kStrideW, kInputSize, kSize>,
            dim3(grid_size), dim3(kBlockSize), (void **)args, 0);

        CUDA_POST_KERNEL_CHECK;
    };
};