#pragma once
#include "net/lstm_cell_net_100_10/WavefrontParams.h"
#include <stdio.h>

enum LstmScaleParams {
    kColumsPerBlock32 = 32,
    kColumsPerBlock64 = 64,
    kThreadNumPerBlock256 = 256,
    kThreadNumPerBlock128 = 128,
    kHiddenSize256 = 256,
    kInputSize256 = 256,
    kCellNumber10 = 10,
    kSeq2SeqEncodeCellNumber8 = 8,
    kSeq2SeqEncodeTimestep100 = 100,
    kSeq2SeqDecodeCellNumber4 = 4,
    kSeq2SeqDecodeTimestep30 = 30,
    kHiddenSize128 = 128,
    kInputSize128 = 128,
    kMask7 = 0x07,
    kMask3 = 0x03,
};

#define call_onekernel_naivefuse_fusedsolve(                                            \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,            \
    kInputSize, kThreadNumperBlock, kMask)                                              \
    {                                                                                   \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_slove_copy_paste< \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(              \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,             \
            output + kOutputOffset);                                                    \
    }

#define call_onekernel_naivefuse_fusedsolve_fusedcompute(                      \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,   \
    kInputSize, kThreadNumperBlock, kMask)                                     \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_fusedsolve_fusedcompute<               \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(     \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,    \
            output + kOutputOffset);                                           \
    }

#define call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw(                \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,   \
    kInputSize, kThreadNumperBlock, kMask)                                     \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_with_adduw_global<                     \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(     \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,    \
            output + kOutputOffset);                                           \
    }

#define call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell( \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,      \
    kInputSize, kThreadNumperBlock, kMask)                                        \
    {                                                                             \
        onekernel_fuse_opt_v2_no_float4_with_adduw_global_16blocks_eachcell<      \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(        \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,       \
            output + kOutputOffset);                                              \
    }

#define call_onekernel_compute_naivefuse(                                         \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,      \
    kInputSize, kThreadNumperBlock, kMask)                                        \
    {                                                                             \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_copy_paste< \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(        \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,       \
            output + kOutputOffset);                                              \
    }

#define call_onekernel_solve_naivefuse(                                        \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,   \
    kInputSize, kThreadNumperBlock, kMask)                                     \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_solve_copy_paste<      \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(     \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,    \
            output + kOutputOffset);                                           \
    }

#define call_onekernel_compute_fusedcompute(                                   \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,   \
    kInputSize, kThreadNumperBlock, kMask)                                     \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh<         \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(     \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,    \
            output + kOutputOffset);                                           \
    }

#define call_onekernel_solve_fusedcompute(                                     \
    kInputOffset, kModelOffset, kOutputOffset, kColumsPerBlock, kHiddenSize,   \
    kInputSize, kThreadNumperBlock, kMask)                                     \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_solve_copy_paste<      \
            kColumsPerBlock, kHiddenSize, kInputSize, kThreadNumperBlock>(     \
            blockIdx.x & kMask, input + kInputOffset, model + kModelOffset,    \
            output + kOutputOffset);                                           \
    }

__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_with_adduw_global_16blocks_eachcell(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    if (lane_id > 15)
        return;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[0] = fma(model->weight_us[0][kStart], data2, temp1[0]);
        temp1[1] = fma(model->weight_us[1][kStart], data2, temp1[1]);
        temp1[2] = fma(model->weight_us[2][kStart], data2, temp1[2]);
        temp1[3] = fma(model->weight_us[3][kStart], data2, temp1[3]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
    __syncthreads();
    if (warp_id == 0) {

        float x, y, z, w;
        x = model->temp[0][colOffset] + model->biass[0][colOffset];
        y = model->temp[1][colOffset] + model->biass[1][colOffset];
        z = model->temp[2][colOffset] + model->biass[2][colOffset];
        w = model->temp[3][colOffset] + model->biass[3][colOffset];

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void onekernel_fuse_opt_v2_no_float4_with_adduw_global(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[0] = fma(model->weight_us[0][kStart], data2, temp1[0]);
        temp1[1] = fma(model->weight_us[1][kStart], data2, temp1[1]);
        temp1[2] = fma(model->weight_us[2][kStart], data2, temp1[2]);
        temp1[3] = fma(model->weight_us[3][kStart], data2, temp1[3]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
    __syncthreads();
    if (warp_id == 0) {

        float x, y, z, w;
        x = model->temp[0][colOffset] + model->biass[0][colOffset];
        y = model->temp[1][colOffset] + model->biass[1][colOffset];
        z = model->temp[2][colOffset] + model->biass[2][colOffset];
        w = model->temp[3][colOffset] + model->biass[3][colOffset];

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    __shared__ float4 nndense_output1[32 * 2];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    nndense_output1[lane_id + 32] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();
    atomicAdd(&nndense_output1[lane_id].x, temp1[0]);
    atomicAdd(&nndense_output1[lane_id].y, temp1[1]);
    atomicAdd(&nndense_output1[lane_id].z, temp1[2]);
    atomicAdd(&nndense_output1[lane_id].w, temp1[3]);
    atomicAdd(&nndense_output1[lane_id + 32].x, temp1[4]);
    atomicAdd(&nndense_output1[lane_id + 32].y, temp1[5]);
    atomicAdd(&nndense_output1[lane_id + 32].z, temp1[6]);
    atomicAdd(&nndense_output1[lane_id + 32].w, temp1[7]);
    __syncthreads();
    if (warp_id == 0) {

        float x, y, z, w;
        // float4 bias_t = model->bias[colOffset];
        x = nndense_output1[lane_id].x + nndense_output1[lane_id + 32].x +
            model->biass[0][colOffset];
        y = nndense_output1[lane_id].y + nndense_output1[lane_id + 32].y +
            model->biass[1][colOffset];
        z = nndense_output1[lane_id].z + nndense_output1[lane_id + 32].z +
            model->biass[2][colOffset];
        w = nndense_output1[lane_id].w + nndense_output1[lane_id + 32].w +
            model->biass[3][colOffset];

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_with_adduw_global_compute_wi_uh(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[0] = fma(model->weight_us[0][kStart], data2, temp1[0]);
        temp1[1] = fma(model->weight_us[1][kStart], data2, temp1[1]);
        temp1[2] = fma(model->weight_us[2][kStart], data2, temp1[2]);
        temp1[3] = fma(model->weight_us[3][kStart], data2, temp1[3]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {
    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    model->temp[4][colOffset] = 0.0;
    model->temp[5][colOffset] = 0.0;
    model->temp[6][colOffset] = 0.0;
    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
    atomicAdd(&model->temp[4][colOffset], temp1[4]);
    atomicAdd(&model->temp[5][colOffset], temp1[5]);
    atomicAdd(&model->temp[6][colOffset], temp1[6]);
    atomicAdd(&model->temp[7][colOffset], temp1[7]);
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void onekernel_fuse_opt_v2_no_float4_fusedsolve_fusedcompute(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    model->temp[4][colOffset] = 0.0;
    model->temp[5][colOffset] = 0.0;
    model->temp[6][colOffset] = 0.0;
    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
    atomicAdd(&model->temp[4][colOffset], temp1[4]);
    atomicAdd(&model->temp[5][colOffset], temp1[5]);
    atomicAdd(&model->temp[6][colOffset], temp1[6]);
    atomicAdd(&model->temp[7][colOffset], temp1[7]);
    __syncthreads();

    if (warp_id == 0) {

        float x, y, z, w;
        x = model->temp[0][colOffset] + model->temp[4][colOffset] +
            model->biass[0][colOffset];
        y = model->temp[1][colOffset] + model->temp[5][colOffset] +
            model->biass[1][colOffset];
        z = model->temp[2][colOffset] + model->temp[6][colOffset] +
            model->biass[2][colOffset];
        w = model->temp[3][colOffset] + model->temp[7][colOffset] +
            model->biass[3][colOffset];

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_copy_paste_index_by_mul(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[0][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_ws[0][i * kHiddenSize + colOffset],
                       input->input_i[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[1][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_ws[1][i * kHiddenSize + colOffset],
                       input->input_i[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[1][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[2][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_ws[2][i * kHiddenSize + colOffset],
                       input->input_i[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[2][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[3][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_ws[3][i * kHiddenSize + colOffset],
                       input->input_i[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[3][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[4][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_us[0][i * kHiddenSize + colOffset],
                       input->input_h[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[4][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[5][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_us[1][i * kHiddenSize + colOffset],
                       input->input_h[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[5][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[6][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_us[2][i * kHiddenSize + colOffset],
                       input->input_h[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[6][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[7][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int kStart = ROWS * warp_id;
        int kEnd = (warp_id + 1) * ROWS;
        for (int i = kStart; i < kEnd; ++i) {
            temp = fma(model->weight_us[3][i * kHiddenSize + colOffset],
                       input->input_h[i], temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[7][colOffset], temp);
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[0][colOffset] = 0.0;
        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[1][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[1][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[2][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[2][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[3][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[3][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[4][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[4][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[5][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[5][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[6][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[6][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[7][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[7][colOffset], temp);
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_copy_paste_easy_fuse(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {
        const int warp_id_0 = threadIdx.x >> 5;
        const int lane_id_0 = threadIdx.x & 0x1f;
        const int colOffset_0 = blockIdx1.x * kColumsPerBlock + lane_id_0;
        float temp_0 = 0.0000f;
        model->temp[0][colOffset_0] = 0.0;

        const int ROWS_0 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_0 = ROWS_0 * warp_id_0;
        int kStart_0 = vectorRow_0 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_0;
        int kEnd_0 = kStart_0 + ROWS_0 * kHiddenSize;
        for (; kStart_0 < kEnd_0; kStart_0 += kHiddenSize, ++vectorRow_0) {
            const float data = input->input_i[vectorRow_0];
            temp_0 = fma(model->weight_ws[0][kStart_0], data, temp_0);
        }
        //
        const int warp_id_1 = threadIdx.x >> 5;
        const int lane_id_1 = threadIdx.x & 0x1f;
        const int colOffset_1 = blockIdx1.x * kColumsPerBlock + lane_id_1;
        float temp_1 = 0.0000f;
        model->temp[1][colOffset_1] = 0.0;

        const int ROWS_1 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_1 = ROWS_1 * warp_id_1;
        int kStart_1 = vectorRow_1 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_1;
        int kEnd_1 = kStart_1 + ROWS_1 * kHiddenSize;
        for (; kStart_1 < kEnd_1; kStart_1 += kHiddenSize, ++vectorRow_1) {
            const float data = input->input_i[vectorRow_1];
            temp_1 = fma(model->weight_ws[1][kStart_1], data, temp_1);
        }
        //
        const int warp_id_2 = threadIdx.x >> 5;
        const int lane_id_2 = threadIdx.x & 0x1f;
        const int colOffset_2 = blockIdx1.x * kColumsPerBlock + lane_id_2;
        float temp_2 = 0.0000f;
        model->temp[2][colOffset_2] = 0.0;

        const int ROWS_2 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_2 = ROWS_2 * warp_id_2;
        int kStart_2 = vectorRow_2 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_2;
        int kEnd_2 = kStart_2 + ROWS_2 * kHiddenSize;
        for (; kStart_2 < kEnd_2; kStart_2 += kHiddenSize, ++vectorRow_2) {
            const float data = input->input_i[vectorRow_2];
            temp_2 = fma(model->weight_ws[2][kStart_2], data, temp_2);
        }
        //
        const int warp_id_3 = threadIdx.x >> 5;
        const int lane_id_3 = threadIdx.x & 0x1f;
        const int colOffset_3 = blockIdx1.x * kColumsPerBlock + lane_id_3;
        float temp_3 = 0.0000f;
        model->temp[3][colOffset_3] = 0.0;

        const int ROWS_3 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_3 = ROWS_3 * warp_id_3;
        int kStart_3 = vectorRow_3 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_3;
        int kEnd_3 = kStart_3 + ROWS_3 * kHiddenSize;
        for (; kStart_3 < kEnd_3; kStart_3 += kHiddenSize, ++vectorRow_3) {
            const float data = input->input_i[vectorRow_3];
            temp_3 = fma(model->weight_ws[3][kStart_3], data, temp_3);
        }
        //
        const int warp_id_4 = threadIdx.x >> 5;
        const int lane_id_4 = threadIdx.x & 0x1f;
        const int colOffset_4 = blockIdx1.x * kColumsPerBlock + lane_id_4;
        float temp_4 = 0.0000f;
        model->temp[4][colOffset_4] = 0.0;

        const int ROWS_4 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_4 = ROWS_4 * warp_id_4;
        int kStart_4 = vectorRow_4 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_4;
        int kEnd_4 = kStart_4 + ROWS_4 * kHiddenSize;
        for (; kStart_4 < kEnd_4; kStart_4 += kHiddenSize, ++vectorRow_4) {
            const float data = input->input_h[vectorRow_4];
            temp_4 = fma(model->weight_us[0][kStart_4], data, temp_4);
        }
        //
        const int warp_id_5 = threadIdx.x >> 5;
        const int lane_id_5 = threadIdx.x & 0x1f;
        const int colOffset_5 = blockIdx1.x * kColumsPerBlock + lane_id_5;
        float temp_5 = 0.0000f;
        model->temp[5][colOffset_5] = 0.0;

        const int ROWS_5 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_5 = ROWS_5 * warp_id_5;
        int kStart_5 = vectorRow_5 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_5;
        int kEnd_5 = kStart_5 + ROWS_5 * kHiddenSize;
        for (; kStart_5 < kEnd_5; kStart_5 += kHiddenSize, ++vectorRow_5) {
            const float data = input->input_h[vectorRow_5];
            temp_5 = fma(model->weight_us[1][kStart_5], data, temp_5);
        }
        //
        const int warp_id_6 = threadIdx.x >> 5;
        const int lane_id_6 = threadIdx.x & 0x1f;
        const int colOffset_6 = blockIdx1.x * kColumsPerBlock + lane_id_6;
        float temp_6 = 0.0000f;
        model->temp[6][colOffset_6] = 0.0;

        const int ROWS_6 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_6 = ROWS_6 * warp_id_6;
        int kStart_6 = vectorRow_6 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_6;
        int kEnd_6 = kStart_6 + ROWS_6 * kHiddenSize;
        for (; kStart_6 < kEnd_6; kStart_6 += kHiddenSize, ++vectorRow_6) {
            const float data = input->input_h[vectorRow_6];
            temp_6 = fma(model->weight_us[2][kStart_6], data, temp_6);
        }
        //
        const int warp_id_7 = threadIdx.x >> 5;
        const int lane_id_7 = threadIdx.x & 0x1f;
        const int colOffset_7 = blockIdx1.x * kColumsPerBlock + lane_id_7;
        float temp_7 = 0.0000f;
        model->temp[7][colOffset_7] = 0.0;

        const int ROWS_7 = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow_7 = ROWS_7 * warp_id_7;
        int kStart_7 = vectorRow_7 * kHiddenSize +
                       blockIdx1.x * kColumsPerBlock + lane_id_7;
        int kEnd_7 = kStart_7 + ROWS_7 * kHiddenSize;
        for (; kStart_7 < kEnd_7; kStart_7 += kHiddenSize, ++vectorRow_7) {
            const float data = input->input_h[vectorRow_7];
            temp_7 = fma(model->weight_us[3][kStart_7], data, temp_7);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset_0], temp_0);
        atomicAdd(&model->temp[1][colOffset_1], temp_1);
        atomicAdd(&model->temp[2][colOffset_2], temp_2);
        atomicAdd(&model->temp[3][colOffset_3], temp_3);
        atomicAdd(&model->temp[4][colOffset_4], temp_4);
        atomicAdd(&model->temp[5][colOffset_5], temp_5);
        atomicAdd(&model->temp[6][colOffset_6], temp_6);
        atomicAdd(&model->temp[7][colOffset_7], temp_7);
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_solve_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;

    if (warp_id == 0) {
        float x, y, z, w;
        x = model->temp[0][colOffset] + model->temp[4][colOffset] +
            model->biass[0][colOffset];
        y = model->temp[1][colOffset] + model->temp[5][colOffset] +
            model->biass[1][colOffset];
        z = model->temp[2][colOffset] + model->temp[6][colOffset] +
            model->biass[2][colOffset];
        w = model->temp[3][colOffset] + model->temp[7][colOffset] +
            model->biass[3][colOffset];

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_slove_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[0][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[1][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[1][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[2][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[2][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[3][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[3][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[4][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[4][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[5][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[5][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[6][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[6][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;
        float temp = 0.0000f;
        model->temp[7][colOffset] = 0.0;

        const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * kHiddenSize + blockIdx1.x * kColumsPerBlock + lane_id;
        int kEnd = kStart + ROWS * kHiddenSize;
        for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[7][colOffset], temp);
    }

    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * kColumsPerBlock + lane_id;

        if (warp_id == 0) {

            float x, y, z, w;
            x = model->temp[0][colOffset] + model->temp[4][colOffset] +
                model->biass[0][colOffset];
            y = model->temp[1][colOffset] + model->temp[5][colOffset] +
                model->biass[1][colOffset];
            z = model->temp[2][colOffset] + model->temp[6][colOffset] +
                model->biass[2][colOffset];
            w = model->temp[3][colOffset] + model->temp[7][colOffset] +
                model->biass[3][colOffset];

            x = sigmoid(x);
            y = tanh(y);
            w = sigmoid(w);
            z = sigmoid(z + 1.0000f) * output->state_c[colOffset];
            output->state_c[colOffset] = fma(x, y, z);
            output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
        }
    }
}
