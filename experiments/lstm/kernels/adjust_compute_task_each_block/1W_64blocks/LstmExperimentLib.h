#include "net/lstm_100_10_experiment/WavefrontParams.h"
#include <stdio.h>
#define COLUMNS_PER_BLOCK 32 // one block compute 32 colums
#define THREAD_NUMS_PER_BLOCK 256
#define HIDDENSIZE 256
#define INPUTSIZE HIDDENSIZE
#define CELL_NUM 10
#define NUM_LAYER 100
#define call_onekernel(cell, step)                                             \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_with_adduw_global(                     \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi(cell, step)                                  \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi(            \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_uh(cell, step)                                  \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_uh(            \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi_uh_0(cell, step)                             \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_0(       \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi_uh_1(cell, step)                             \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_1(       \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi_uh_2(cell, step)                             \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_2(       \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi_uh_3(cell, step)                             \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_3(       \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_solve(cell, step)                                       \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_solve_copy_paste(      \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }
#define call_onekernel_compute_2_wi_uh_0(cell, step)                           \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_2_wi_uh_0(     \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_2_wi_uh_1(cell, step)                           \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_2_wi_uh_1(     \
            blockIdx.x & 0x7, input + step * CELL_NUM + cell, model + cell,    \
            output + step * CELL_NUM + cell);                                  \
    }

#define call_onekernel_compute_wi_0(cell, step)                                \
    {                                                                          \
        compute_wi_0(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_wi_1(cell, step)                                \
    {                                                                          \
        compute_wi_1(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_wi_2(cell, step)                                \
    {                                                                          \
        compute_wi_2(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_wi_3(cell, step)                                \
    {                                                                          \
        compute_wi_3(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }

#define call_onekernel_compute_uh_0(cell, step)                                \
    {                                                                          \
        compute_uh_0(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_uh_1(cell, step)                                \
    {                                                                          \
        compute_uh_1(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_uh_2(cell, step)                                \
    {                                                                          \
        compute_uh_2(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }
#define call_onekernel_compute_uh_3(cell, step)                                \
    {                                                                          \
        compute_uh_3(blockIdx.x & 0x7, input + step * CELL_NUM + cell,         \
                     model + cell, output + step * CELL_NUM + cell);           \
    }

__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    #pragma unroll 8
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_uh(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[4][colOffset] = 0.0;
    model->temp[5][colOffset] = 0.0;
    model->temp[6][colOffset] = 0.0;
    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    #pragma unroll 8
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data2 = input->input_h[vectorRow];

        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();
    atomicAdd(&model->temp[4][colOffset], temp1[4]);
    atomicAdd(&model->temp[5][colOffset], temp1[5]);
    atomicAdd(&model->temp[6][colOffset], temp1[6]);
    atomicAdd(&model->temp[7][colOffset], temp1[7]);
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_0(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;

    model->temp[4][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);

        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);

    atomicAdd(&model->temp[4][colOffset], temp1[4]);
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_1(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[1][colOffset] = 0.0;

    model->temp[5][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);

        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
    }
    __syncthreads();
    atomicAdd(&model->temp[1][colOffset], temp1[1]);

    atomicAdd(&model->temp[5][colOffset], temp1[5]);
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_2(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {
    //
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[2][colOffset] = 0.0;

    model->temp[6][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];

        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);

        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
    }
    __syncthreads();

    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[6][colOffset], temp1[6]);
}

__device__ static void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh_3(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[3][colOffset] = 0.0;

    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    #pragma unroll 8
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];

        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);

        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();

    atomicAdd(&model->temp[3][colOffset], temp1[3]);

    atomicAdd(&model->temp[7][colOffset], temp1[7]);
}

__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_2_wi_uh_0(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;

    model->temp[4][colOffset] = 0.0;
    model->temp[5][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);

        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);

    atomicAdd(&model->temp[4][colOffset], temp1[4]);
    atomicAdd(&model->temp[5][colOffset], temp1[5]);
}
__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_2_wi_uh_1(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;

    model->temp[6][colOffset] = 0.0;
    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        const float data2 = input->input_h[vectorRow];

        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);

        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();

    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);

    atomicAdd(&model->temp[6][colOffset], temp1[6]);
    atomicAdd(&model->temp[7][colOffset], temp1[7]);
}

__device__ static void
onekernel_fuse_opt_v2_no_float4_no_adduw_global_solve_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;

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
__device__ static void compute_wi_0(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        temp1[0] = fma(model->weight_ws[0][kStart], data, temp1[0]);
    }
    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
}

__device__ static void compute_wi_1(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[1][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        temp1[1] = fma(model->weight_ws[1][kStart], data, temp1[1]);
    }
    __syncthreads();
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
}

__device__ static void compute_wi_2(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[2][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];

        temp1[2] = fma(model->weight_ws[2][kStart], data, temp1[2]);
    }
    __syncthreads();

    atomicAdd(&model->temp[2][colOffset], temp1[2]);
}

__device__ static void compute_wi_3(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[3][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data = input->input_i[vectorRow];
        temp1[3] = fma(model->weight_ws[3][kStart], data, temp1[3]);
    }
    __syncthreads();

    atomicAdd(&model->temp[3][colOffset], temp1[3]);
}

__device__ static void compute_uh_0(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[4][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data2 = input->input_h[vectorRow];

        temp1[4] = fma(model->weight_us[0][kStart], data2, temp1[4]);
    }
    __syncthreads();
    atomicAdd(&model->temp[4][colOffset], temp1[4]);
}

__device__ static void compute_uh_1(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[5][colOffset] = 0.0;

    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data2 = input->input_h[vectorRow];

        temp1[5] = fma(model->weight_us[1][kStart], data2, temp1[5]);
    }
    __syncthreads();
    atomicAdd(&model->temp[5][colOffset], temp1[5]);
}

__device__ static void compute_uh_2(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[6][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
        const float data2 = input->input_h[vectorRow];

        temp1[6] = fma(model->weight_us[2][kStart], data2, temp1[6]);
    }
    __syncthreads();

    atomicAdd(&model->temp[6][colOffset], temp1[6]);
}

__device__ static void compute_uh_3(dim3 blockIdx1,
                             WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    model->temp[7][colOffset] = 0.0;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        // int idx_weight = kStart + i * 256;
         
        const float data2 = input->input_h[vectorRow];
        temp1[7] = fma(model->weight_us[3][kStart], data2, temp1[7]);
    }
    __syncthreads();
    atomicAdd(&model->temp[7][colOffset], temp1[7]);
}