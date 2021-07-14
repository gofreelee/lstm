#pragma once
#include "cuda_ops.h"
#include "utils.h"

template <int kBatchSize, int kInSize, int kOutSize, bool kIsBias,
          bool kFuseRelu>
class Linear {
  public:
    typedef LinearState<kBatchSize, kInSize, kOutSize, kIsBias> StateType;
    static void Forward(const Tensor<kBatchSize * kInSize> *input,
                        StateType *state) {

        dim3 dim_block(kTileSize, kTileSize);

        dim3 dim_grid((kOutSize + kTileSize - 1) / kTileSize,
                      (kBatchSize + kTileSize - 1) / kTileSize, 1);

        void *weight_ptr = &(state->weight);
        void *output_ptr = &(state->output);

        if constexpr (kIsBias) {
            void *bias_ptr = &(state->bias);
            void *args[] = {&input, &weight_ptr, &output_ptr, &bias_ptr};
            cudaLaunchKernel(
                (const void *)operator_fuse_matmul_bias_relu_h<
                    kBatchSize, kInSize, kOutSize, 0, 1, 1, kFuseRelu>,
                dim_grid, dim_block, (void **)args, kTileSize * kTileSize * 2);
        } else {
            void *args[] = {&input, &weight_ptr, &output_ptr};
            cudaLaunchKernel(
                (const void *)operator_fuse_matmul_relu_h<
                    kBatchSize, kInSize, kOutSize, 0, 1, 1, kFuseRelu>,
                dim_grid, dim_block, (void **)args, kTileSize * kTileSize * 2);
        }
        CUDA_POST_KERNEL_CHECK;
    }
};