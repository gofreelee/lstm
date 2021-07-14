#pragma once
#include "cuda_ops.h"
#include "utils.h"

// High Performance Convolutional Neural Networks for Document Processing
// https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf

template <int kBatchSize, int kHeight, int kWidth, int kChannelIn,
          int kChannelOut, int kKernelH, int kKernelW, int kPadH, int kPadW,
          int kStrideH, int kStrideW, bool kIsBias, bool kFuseBN,
          bool kFuseRelu, int kGroupSize>
class Conv {
  public:
    enum {
        kInputSize = kBatchSize * kChannelIn * kHeight * kWidth,
        kPoolHeight = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1,
        kPoolWidth = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1,
        kOutputSize = kBatchSize * kChannelOut * kPoolHeight * kPoolWidth,
        kFilterSize =
            kChannelOut * kChannelIn / kGroupSize * kKernelH * kKernelW,
        kBiasSize = kChannelOut * kIsBias,
        kColNum = kChannelIn * kPoolHeight * kPoolWidth,
        kColSize = kBatchSize * kColNum * kKernelH * kKernelW,
    };

    typedef ConvState<kOutputSize, kChannelOut, kFilterSize, kColSize,
                      kBiasSize, kIsBias, kFuseBN>
        StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        // im2col
        // [kBatchSize*(C_in*k_h*k_w)*(height_col * width_col)]
        dim3 im2col_dim_grid((kColNum / kGroupSize + kBlockSize - 1) /
                                 kBlockSize,
                             kBatchSize * kGroupSize);

        void *col_ptr = &state->col;
        void *im2col_args[] = {(void *)&input, &col_ptr};
        cudaLaunchKernel(
            (const void *)im2col_h<kBatchSize * kGroupSize, kHeight, kWidth,
                                   kChannelIn / kGroupSize, kChannelOut,
                                   kKernelH, kKernelW, kPadH, kPadW, kStrideH,
                                   kStrideW, kIsBias, kInputSize, kColSize>,
            im2col_dim_grid, dim3(kBlockSize), (void **)im2col_args, 0);

        CUDA_POST_KERNEL_CHECK;

        // Y = F * col
        // [C_out*(C_in*k_h*k_w)] * [kBatchSize *
        // (C_in*k_h*k_w)*(height_col*width_col)] = [kBatchSize * kChannelOut *
        // (height_col * width_col)]

        dim3 dim_block(kTileSize, kTileSize);

        constexpr int kHeightOut =
            (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
        constexpr int kWidthOut =
            (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;

        constexpr int kMatmulHeight = kChannelOut / kGroupSize;
        constexpr int kMatmulK = kChannelIn / kGroupSize * kKernelH * kKernelW;
        constexpr int kMatmulWidth = kWidthOut * kHeightOut;

        dim3 matmul_dim_grid((kMatmulWidth + kTileSize - 1) / kTileSize,
                             (kMatmulHeight + kTileSize - 1) / kTileSize,
                             kBatchSize * kGroupSize);

        void *filter_ptr = &state->filter;
        void *output_ptr = &state->output;

        if constexpr (kFuseBN) {
            static_assert(kIsBias == false);
            void *bn_ptr = &state->bn_param;
            void *fuse_args[] = {&filter_ptr, &col_ptr, &output_ptr, &bn_ptr};
            cudaLaunchKernel(
                (const void *)operator_fuse_conv_bn_relu_h<
                    kMatmulHeight, kMatmulK, kMatmulWidth, 1, 1, kBatchSize,
                    kChannelOut, kHeightOut, kWidthOut, kFuseRelu, kGroupSize>,
                matmul_dim_grid, dim_block, (void **)fuse_args,
                kTileSize * kTileSize * 2);
        } else {
            if constexpr (kIsBias) {
                void *bias_ptr = &state->bias;
                void *matmul_args[] = {&filter_ptr, &col_ptr, &output_ptr,
                                       &bias_ptr};
                cudaLaunchKernel((const void *)operator_fuse_conv_bias_relu_h<
                                     kMatmulHeight, kMatmulK, kMatmulWidth, 1,
                                     1, kBatchSize, kFuseRelu, kGroupSize>,
                                 matmul_dim_grid, dim_block,
                                 (void **)matmul_args,
                                 kTileSize * kTileSize * 2);
            } else {
                void *matmul_args[] = {&filter_ptr, &col_ptr, &output_ptr};
                cudaLaunchKernel((const void *)operator_fuse_matmul_relu_h<
                                     kMatmulHeight, kMatmulK, kMatmulWidth, 1,
                                     1, kBatchSize, kFuseRelu, kGroupSize>,
                                 matmul_dim_grid, dim_block,
                                 (void **)matmul_args,
                                 kTileSize * kTileSize * 2);
            }
        }

        CUDA_POST_KERNEL_CHECK;
    }
};
