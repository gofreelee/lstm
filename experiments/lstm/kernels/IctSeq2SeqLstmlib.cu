#include "net/seq2seq_net/Seq2SeqArgs.h"

#define COLUMNS_PER_BLOCK 32 // one block compute 32 colums
#define THREAD_NUMS_PER_BLOCK 256
#define HIDDENSIZE 256
#define INPUTSIZE HIDDENSIZE
#define HIDDENSIZE_128 128
#define INPUTSIZE_128 128
#define CELL_NUM_ENC 8
#define CELL_NUM_DEC 4
#define TIME_STEP_ENC 100
#define TIME_STEP_DEC 30

#define call_onekernel_seq2seq_enc(cell, step)                                              \
    {                                                                                       \
        onekernel_fuse_opt_v2_128_no_float4_with_adduw_global( \
            blockIdx.x & 0x3, input + step * CELL_NUM_ENC + cell,                           \
            model + cell, output + step * CELL_NUM_ENC + cell);                             \
    }

#define call_onekernel_seq2seq_dec(cell, step)                                              \
    {                                                                                       \
        onekernel_fuse_opt_v2_128_no_float4_with_adduw_global( \
            blockIdx.x & 0x3,                                                               \
            input + step * CELL_NUM_DEC + cell + CELL_NUM_ENC * TIME_STEP_ENC,              \
            model + cell + CELL_NUM_ENC,                                                    \
            output + step * CELL_NUM_DEC + cell +                                           \
                CELL_NUM_ENC * TIME_STEP_ENC);                                              \
    }

#define call_onekernel_compute_seq2seq_enc(cell, step)                         \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh(         \
            blockIdx.x & 0x3, input + step * CELL_NUM_ENC + cell,              \
            model + cell, output + step * CELL_NUM_ENC + cell);                \
    }

#define call_onekernel_compute_seq2seq_dec(cell, step)                         \
    {                                                                          \
        onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh(         \
            blockIdx.x & 0x3,                                                  \
            input + step * CELL_NUM_DEC + cell + CELL_NUM_ENC * TIME_STEP_ENC, \
            model + cell + CELL_NUM_ENC,                                       \
            output + step * CELL_NUM_DEC + cell +                              \
                CELL_NUM_ENC * TIME_STEP_ENC);                                 \
    }

#define call_onekernel_solve_seq2seq_enc(cell, step)                           \
    {                                                                          \
        onekernel_fuse_opt_v2_128_no_float4_no_adduw_global_solve_copy_paste(  \
            blockIdx.x & 0x3, input + step * CELL_NUM_ENC + cell,              \
            model + cell, output + step * CELL_NUM_ENC + cell);                \
    }

#define call_onekernel_solve_seq2seq_dec(cell, step)                           \
    {                                                                          \
        onekernel_fuse_opt_v2_128_no_float4_no_adduw_global_solve_copy_paste(  \
            blockIdx.x & 0x3,                                                  \
            input + step * CELL_NUM_DEC + cell + CELL_NUM_ENC * TIME_STEP_ENC, \
            model + cell + CELL_NUM_ENC,                                       \
            output + step * CELL_NUM_DEC + cell +                              \
                CELL_NUM_ENC * TIME_STEP_ENC);                                 \
    }

__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void gemv_128(const float *__restrict__ input,
                         const float *__restrict__ weight,
                         float *__restrict__ output) {

    __shared__ float nndense_output[COLUMNS_PER_BLOCK];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    nndense_output[lane_id] = 0.0000f;
    float temp = 0.0000f;
    const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE_128 + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE_128;
    __syncthreads();
    for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
        const float data = input[vectorRow];
        temp = fma(weight[kStart], data, temp);
    }

    atomicAdd(&nndense_output[lane_id], temp);
    __syncthreads();
    if (warp_id == 0)
        output[colOffset] = nndense_output[lane_id];
}

__global__ void solve_128(float *t00, float *t01, float *b0, float *t10,
                          float *t11, float *b1, float *t20, float *t21,
                          float *b2, float *t30, float *t31, float *b3,
                          float *state_c, float *state_h) {
    const int idx = threadIdx.x;
    float x = t00[idx] + t01[idx] + b0[idx];
    float y = t10[idx] + t11[idx] + b1[idx];
    float z = t20[idx] + t21[idx] + b2[idx];
    float w = t30[idx] + t31[idx] + b3[idx];
    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(z) * state_c[idx];
    state_c[idx] = fma(x, y, z);
    state_h[idx] = (tanh(state_c[idx])) * w;
}

__global__ void gemv(const float *__restrict__ input,
                     const float *__restrict__ weight,
                     float *__restrict__ output) {
    __shared__ float nndense_output[COLUMNS_PER_BLOCK];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    nndense_output[lane_id] = 0.0000f;
    float temp = 0.0000f;
    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        const float data = input[vectorRow];
        temp = fma(weight[kStart], data, temp);
    }

    atomicAdd(&nndense_output[lane_id], temp);
    __syncthreads();
    if (warp_id == 0)
        output[colOffset] = nndense_output[lane_id];
}

__global__ void solve(float *t00, float *t01, float *b0, float *t10, float *t11,
                      float *b1, float *t20, float *t21, float *b2, float *t30,
                      float *t31, float *b3, float *state_c, float *state_h) {
    const int idx = threadIdx.x;
    float x = t00[idx] + t01[idx] + b0[idx];
    float y = t10[idx] + t11[idx] + b1[idx];
    float z = t20[idx] + t21[idx] + b2[idx];
    float w = t30[idx] + t31[idx] + b3[idx];
    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(z + 1.0000f) * state_c[idx];
    state_c[idx] = fma(x, y, z);
    state_h[idx] = (tanh(state_c[idx])) * w;
    // sigmoid(z) + 1.0000f
}

// 一次算w0 ~ w3 和 input 的四个gemv，  或者u0 ~ u3 和 state_h的gemv
__global__ void gem4v(const float *__restrict__ input,
                      const float4 *__restrict__ weight,
                      float4 *__restrict__ output) {
    __shared__ float4 nndense_output[COLUMNS_PER_BLOCK];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    nndense_output[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    float temp[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        const float data = input[vectorRow];
        float4 res = weight[kStart];
        temp[0] = fma(res.x, data, temp[0]);
        temp[1] = fma(res.y, data, temp[1]);
        temp[2] = fma(res.z, data, temp[2]);
        temp[3] = fma(res.w, data, temp[3]);
    }
    //__syncthreads();

    atomicAdd(&nndense_output[lane_id].x, temp[0]);
    atomicAdd(&nndense_output[lane_id].y, temp[1]);
    atomicAdd(&nndense_output[lane_id].z, temp[2]);
    atomicAdd(&nndense_output[lane_id].w, temp[3]);
    __syncthreads();
    if (warp_id == 0) {
        output[colOffset] = nndense_output[lane_id];
    }
}

__global__ void solve_gem4v_res(float4 *__restrict__ wi,
                                float4 *__restrict__ uh, float4 *bias,
                                float *state_c, float *state_h) {
    const int idx = threadIdx.x;
    float x, y, z, w;
    x = wi[idx].x + uh[idx].x + bias[idx].x;
    y = wi[idx].y + uh[idx].y + bias[idx].y;
    z = wi[idx].z + uh[idx].z + bias[idx].z;
    w = wi[idx].w + uh[idx].w + bias[idx].w;

    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(sigmoid(z) + 1.0000f) * state_c[idx];
    state_c[idx] = fma(x, y, z);
    state_h[idx] = (tanh(state_c[idx])) * w;
}

__device__ void
onekernel_fuse_opt_v2_128(dim3 blockIdx1, WaveInputParams *__restrict__ input,
                          WaveModelParams *__restrict__ model,
                          WaveOutputParams *__restrict__ output) {

    __shared__ float4 nndense_output1[COLUMNS_PER_BLOCK];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * 32 + lane_id;
    nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = 16; // INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32); // 16
    int vectorRow = ROWS * warp_id; //
    int kStart = vectorRow * 128 + blockIdx1.x * 32 + lane_id;
    int kEnd = kStart + ROWS * 128;
    for (; kStart < kEnd; kStart += 128, ++vectorRow) {
        // printf("%d   %d  %d    blockid: %d  warpid : %d   laneid : %d\n",
        // vectorRow, kStart, kEnd, blockIdx1.x, warp_id, lane_id); printf("%d
        // \n", input->input_i);
        const float data = input->input_i[vectorRow];

        float4 res = model->weight_w[kStart];

        temp1[0] = fma(res.x, data, temp1[0]);
        temp1[1] = fma(res.y, data, temp1[1]);
        temp1[2] = fma(res.z, data, temp1[2]);
        temp1[3] = fma(res.w, data, temp1[3]);
        const float data2 = input->input_h[vectorRow];

        float4 res2 = model->weight_u[kStart];

        temp1[0] = fma(res2.x, data2, temp1[0]);
        temp1[1] = fma(res2.y, data2, temp1[1]);
        temp1[2] = fma(res2.z, data2, temp1[2]);
        temp1[3] = fma(res2.w, data2, temp1[3]);
    }

    __syncthreads();
    atomicAdd(&nndense_output1[lane_id].x, temp1[0]);
    atomicAdd(&nndense_output1[lane_id].y, temp1[1]);
    atomicAdd(&nndense_output1[lane_id].z, temp1[2]);
    atomicAdd(&nndense_output1[lane_id].w, temp1[3]);
    __syncthreads();
    if (warp_id == 0) {
        float x, y, z, w;
        float4 bias_t = model->bias[colOffset];
        x = nndense_output1[lane_id].x + bias_t.x;
        y = nndense_output1[lane_id].y + bias_t.y;
        z = nndense_output1[lane_id].z + bias_t.z;
        w = nndense_output1[lane_id].w + bias_t.w;

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

__device__ void onekernel_fuse_opt_v2_128_no_float4_with_adduw_global(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE_128 + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE_128;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
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
        z = sigmoid(z) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

__device__ void
onekernel_fuse_opt_v2_128_global(dim3 blockIdx1,
                                 WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * 32 + lane_id;
    model->temp[0][colOffset] = 0.0;
    model->temp[1][colOffset] = 0.0;
    model->temp[2][colOffset] = 0.0;
    model->temp[3][colOffset] = 0.0;
    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = 16; // INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32); // 16
    int vectorRow = ROWS * warp_id; //
    int kStart = vectorRow * 128 + blockIdx1.x * 32 + lane_id;
    int kEnd = kStart + ROWS * 128;
    for (; kStart < kEnd; kStart += 128, ++vectorRow) {
        // printf("%d   %d  %d    blockid: %d  warpid : %d   laneid : %d\n",
        // vectorRow, kStart, kEnd, blockIdx1.x, warp_id, lane_id); printf("%d
        // \n", input->input_i);
        const float data = input->input_i[vectorRow];

        float4 res = model->weight_w[kStart];

        temp1[0] = fma(res.x, data, temp1[0]);
        temp1[1] = fma(res.y, data, temp1[1]);
        temp1[2] = fma(res.z, data, temp1[2]);
        temp1[3] = fma(res.w, data, temp1[3]);
        const float data2 = input->input_h[vectorRow];

        float4 res2 = model->weight_u[kStart];

        temp1[0] = fma(res2.x, data2, temp1[0]);
        temp1[1] = fma(res2.y, data2, temp1[1]);
        temp1[2] = fma(res2.z, data2, temp1[2]);
        temp1[3] = fma(res2.w, data2, temp1[3]);
    }

    __syncthreads();
    atomicAdd(&model->temp[0][colOffset], temp1[0]);
    atomicAdd(&model->temp[1][colOffset], temp1[1]);
    atomicAdd(&model->temp[2][colOffset], temp1[2]);
    atomicAdd(&model->temp[3][colOffset], temp1[3]);
    __syncthreads();
    if (warp_id == 0) {
        float x, y, z, w;
        float4 bias_t = model->bias[colOffset];
        x = model->temp[0][colOffset] + bias_t.x;
        y = model->temp[1][colOffset] + bias_t.y;
        z = model->temp[2][colOffset] + bias_t.z;
        w = model->temp[3][colOffset] + bias_t.w;

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

__device__ void
onekernel_fuse_opt_v2_128_no_float4_no_adduw_global_compute_wi_uh_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[0][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[1][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[1][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[2][colOffset] = 0.0;

        const int ROWS = HIDDENSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[2][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[3][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[3][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[4][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[4][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[5][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[5][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[6][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[6][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[7][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[7][colOffset], temp);
    }
}

__device__ void onekernel_fuse_opt_v2_no_float4_no_adduw_global_compute_wi_uh(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    //__shared__ float4 nndense_output1[32];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
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

    const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE_128 + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE_128;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
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

__device__ void
onekernel_fuse_opt_v2_128_no_float4_no_adduw_global_compute_wi_uh_slove_copy_paste(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {

    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[0][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[0][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[1][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[1][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[2][colOffset] = 0.0;

        const int ROWS = HIDDENSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[2][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[3][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_i[vectorRow];
            temp = fma(model->weight_ws[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[3][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[4][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[0][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[4][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[5][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[1][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[5][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[6][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[2][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[6][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        float temp = 0.0000f;
        model->temp[7][colOffset] = 0.0;

        const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart = vectorRow * HIDDENSIZE_128 +
                     blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE_128;
        for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
            const float data = input->input_h[vectorRow];
            temp = fma(model->weight_us[3][kStart], data, temp);
        }
        __syncthreads();
        atomicAdd(&model->temp[7][colOffset], temp);
    }
    if (blockIdx1.x < 8) {
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
            z = sigmoid(z) * output->state_c[colOffset];
            output->state_c[colOffset] = fma(x, y, z);
            output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
        }
    }
}

__device__ void onekernel_fuse_opt_v2_128_no_float4_fusedsolve_fusedcompute(
    dim3 blockIdx1, WaveInputParams *__restrict__ input,
    WaveModelParams *__restrict__ model,
    WaveOutputParams *__restrict__ output) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    // nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    model->temp[0][colOffset] = 0.0000f;
    model->temp[1][colOffset] = 0.0000f;
    model->temp[2][colOffset] = 0.0000f;
    model->temp[3][colOffset] = 0.0000f;
    model->temp[4][colOffset] = 0.0000f;
    model->temp[5][colOffset] = 0.0000f;
    model->temp[6][colOffset] = 0.0000f;
    model->temp[7][colOffset] = 0.0000f;
    float temp1[8] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE_128 / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE_128 + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE_128;
    // This loop was unrolled 4 times in SASS code, I tested other unroll
    // parameters, e.g. 2, 4,6, 8, 4 is the best one
    //#pragma unroll 4
    for (; kStart < kEnd; kStart += HIDDENSIZE_128, ++vectorRow) {
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
    //__shared__ float4 nndense_output1[32];
    __syncthreads();

    if (blockIdx1.x < 8) {
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
            z = sigmoid(z) * output->state_c[colOffset];
            output->state_c[colOffset] = fma(x, y, z);
            output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
        }
    }
}

__device__ void
onekernel_fuse_opt_v2_128_no_float4_no_adduw_global_solve_copy_paste(
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
        z = sigmoid(z) * output->state_c[colOffset];
        output->state_c[colOffset] = fma(x, y, z);
        output->state_h[colOffset] = (tanh(output->state_c[colOffset])) * w;
    }
}

__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave0(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave1(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 1);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave2(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 2);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 1);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave3(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 3);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 2);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 1);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave4(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 4);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 3);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 2);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 1);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave5(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 5);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 4);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 3);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 2);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 1);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave6(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 6);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 5);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 4);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 3);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 2);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 1);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave7(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 7);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 6);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 5);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 4);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 3);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 2);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 1);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave8(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 8);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 7);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 6);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 5);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 4);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 3);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 2);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 1);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave9(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 9);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 8);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 7);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 6);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 5);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 4);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 3);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 2);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave10(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 10);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 9);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 8);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 7);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 6);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 5);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 4);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 3);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave11(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 11);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 10);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 9);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 8);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 7);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 6);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 5);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 4);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave12(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 12);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 11);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 10);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 9);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 8);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 7);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 6);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 5);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave13(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 13);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 12);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 11);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 10);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 9);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 8);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 7);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 6);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave14(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 14);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 13);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 12);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 11);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 10);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 9);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 8);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 7);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave15(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 15);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 14);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 13);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 12);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 11);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 10);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 9);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 8);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave16(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 16);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 15);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 14);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 13);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 12);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 11);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 10);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 9);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave17(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 17);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 16);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 15);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 14);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 13);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 12);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 11);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 10);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave18(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 18);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 17);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 16);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 15);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 14);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 13);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 12);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 11);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave19(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 19);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 18);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 17);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 16);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 15);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 14);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 13);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 12);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave20(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 20);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 19);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 18);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 17);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 16);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 15);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 14);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 13);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave21(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 21);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 20);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 19);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 18);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 17);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 16);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 15);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 14);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave22(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 22);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 21);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 20);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 19);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 18);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 17);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 16);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 15);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave23(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 23);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 22);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 21);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 20);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 19);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 18);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 17);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 16);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave24(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 24);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 23);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 22);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 21);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 20);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 19);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 18);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 17);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave25(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 25);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 24);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 23);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 22);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 21);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 20);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 19);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 18);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave26(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 26);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 25);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 24);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 23);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 22);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 21);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 20);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 19);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave27(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 27);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 26);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 25);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 24);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 23);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 22);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 21);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 20);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave28(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 28);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 27);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 26);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 25);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 24);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 23);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 22);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 21);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave29(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 29);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 28);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 27);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 26);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 25);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 24);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 23);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 22);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave30(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 30);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 29);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 28);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 27);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 26);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 25);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 24);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 23);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave31(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 31);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 30);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 29);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 28);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 27);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 26);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 25);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 24);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave32(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 32);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 31);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 30);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 29);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 28);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 27);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 26);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 25);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave33(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 33);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 32);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 31);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 30);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 29);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 28);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 27);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 26);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave34(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 34);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 33);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 32);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 31);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 30);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 29);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 28);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 27);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave35(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 35);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 34);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 33);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 32);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 31);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 30);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 29);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 28);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave36(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 36);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 35);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 34);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 33);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 32);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 31);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 30);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 29);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave37(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 37);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 36);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 35);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 34);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 33);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 32);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 31);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 30);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave38(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 38);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 37);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 36);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 35);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 34);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 33);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 32);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 31);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave39(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 39);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 38);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 37);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 36);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 35);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 34);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 33);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 32);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave40(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 40);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 39);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 38);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 37);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 36);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 35);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 34);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 33);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave41(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 41);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 40);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 39);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 38);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 37);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 36);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 35);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 34);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave42(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 42);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 41);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 40);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 39);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 38);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 37);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 36);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 35);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave43(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 43);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 42);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 41);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 40);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 39);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 38);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 37);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 36);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave44(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 44);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 43);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 42);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 41);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 40);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 39);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 38);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 37);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave45(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 45);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 44);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 43);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 42);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 41);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 40);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 39);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 38);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave46(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 46);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 45);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 44);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 43);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 42);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 41);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 40);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 39);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave47(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 47);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 46);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 45);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 44);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 43);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 42);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 41);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 40);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave48(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 48);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 47);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 46);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 45);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 44);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 43);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 42);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 41);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave49(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 49);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 48);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 47);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 46);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 45);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 44);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 43);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 42);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave50(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 50);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 49);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 48);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 47);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 46);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 45);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 44);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 43);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave51(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 51);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 50);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 49);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 48);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 47);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 46);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 45);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 44);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave52(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 52);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 51);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 50);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 49);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 48);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 47);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 46);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 45);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave53(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 53);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 52);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 51);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 50);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 49);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 48);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 47);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 46);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave54(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 54);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 53);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 52);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 51);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 50);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 49);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 48);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 47);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave55(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 55);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 54);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 53);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 52);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 51);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 50);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 49);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 48);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave56(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 56);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 55);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 54);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 53);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 52);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 51);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 50);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 49);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave57(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 57);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 56);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 55);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 54);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 53);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 52);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 51);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 50);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave58(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 58);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 57);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 56);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 55);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 54);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 53);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 52);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 51);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave59(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 59);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 58);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 57);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 56);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 55);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 54);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 53);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 52);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave60(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 60);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 59);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 58);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 57);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 56);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 55);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 54);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 53);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave61(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 61);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 60);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 59);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 58);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 57);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 56);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 55);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 54);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave62(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 62);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 61);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 60);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 59);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 58);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 57);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 56);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 55);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave63(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 63);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 62);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 61);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 60);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 59);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 58);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 57);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 56);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave64(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 64);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 63);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 62);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 61);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 60);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 59);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 58);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 57);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave65(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 65);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 64);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 63);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 62);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 61);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 60);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 59);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 58);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave66(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 66);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 65);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 64);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 63);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 62);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 61);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 60);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 59);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave67(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 67);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 66);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 65);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 64);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 63);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 62);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 61);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 60);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave68(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 68);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 67);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 66);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 65);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 64);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 63);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 62);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 61);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave69(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 69);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 68);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 67);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 66);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 65);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 64);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 63);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 62);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave70(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 70);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 69);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 68);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 67);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 66);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 65);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 64);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 63);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave71(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 71);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 70);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 69);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 68);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 67);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 66);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 65);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 64);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave72(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 72);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 71);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 70);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 69);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 68);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 67);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 66);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 65);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave73(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 73);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 72);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 71);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 70);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 69);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 68);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 67);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 66);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave74(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 74);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 73);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 72);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 71);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 70);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 69);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 68);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 67);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave75(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 75);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 74);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 73);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 72);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 71);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 70);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 69);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 68);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave76(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 76);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 75);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 74);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 73);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 72);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 71);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 70);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 69);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave77(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 77);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 76);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 75);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 74);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 73);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 72);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 71);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 70);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave78(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 78);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 77);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 76);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 75);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 74);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 73);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 72);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 71);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave79(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 79);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 78);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 77);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 76);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 75);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 74);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 73);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 72);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave80(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 80);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 79);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 78);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 77);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 76);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 75);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 74);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 73);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave81(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 81);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 80);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 79);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 78);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 77);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 76);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 75);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 74);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave82(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 82);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 81);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 80);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 79);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 78);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 77);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 76);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 75);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave83(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 83);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 82);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 81);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 80);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 79);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 78);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 77);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 76);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave84(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 84);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 83);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 82);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 81);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 80);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 79);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 78);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 77);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave85(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 85);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 84);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 83);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 82);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 81);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 80);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 79);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 78);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave86(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 86);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 85);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 84);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 83);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 82);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 81);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 80);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 79);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave87(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 87);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 86);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 85);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 84);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 83);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 82);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 81);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 80);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave88(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 88);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 87);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 86);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 85);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 84);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 83);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 82);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 81);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave89(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 89);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 88);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 87);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 86);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 85);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 84);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 83);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 82);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave90(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 90);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 89);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 88);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 87);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 86);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 85);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 84);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 83);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave91(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 91);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 90);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 89);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 88);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 87);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 86);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 85);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 84);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave92(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 92);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 91);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 90);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 89);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 88);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 87);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 86);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 85);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave93(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 93);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 92);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 91);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 90);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 89);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 88);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 87);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 86);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave94(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 94);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 93);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 92);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 91);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 90);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 89);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 88);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 87);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave95(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 95);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 94);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 93);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 92);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 91);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 90);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 89);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 88);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave96(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 96);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 95);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 94);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 93);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 92);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 91);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 90);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 89);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave97(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 97);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 96);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 95);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 94);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 93);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 92);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 91);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 90);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave98(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 98);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 97);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 96);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 95);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 94);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 93);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 92);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 91);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave99(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(0, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(1, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(2, 97);
        break;
    case 3:
        call_onekernel_seq2seq_enc(3, 96);
        break;
    case 4:
        call_onekernel_seq2seq_enc(4, 95);
        break;
    case 5:
        call_onekernel_seq2seq_enc(5, 94);
        break;
    case 6:
        call_onekernel_seq2seq_enc(6, 93);
        break;
    case 7:
        call_onekernel_seq2seq_enc(7, 92);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave100(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(1, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(2, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(3, 97);
        break;
    case 3:
        call_onekernel_seq2seq_enc(4, 96);
        break;
    case 4:
        call_onekernel_seq2seq_enc(5, 95);
        break;
    case 5:
        call_onekernel_seq2seq_enc(6, 94);
        break;
    case 6:
        call_onekernel_seq2seq_enc(7, 93);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave101(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(2, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(3, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(4, 97);
        break;
    case 3:
        call_onekernel_seq2seq_enc(5, 96);
        break;
    case 4:
        call_onekernel_seq2seq_enc(6, 95);
        break;
    case 5:
        call_onekernel_seq2seq_enc(7, 94);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave102(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(3, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(4, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(5, 97);
        break;
    case 3:
        call_onekernel_seq2seq_enc(6, 96);
        break;
    case 4:
        call_onekernel_seq2seq_enc(7, 95);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave103(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(4, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(5, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(6, 97);
        break;
    case 3:
        call_onekernel_seq2seq_enc(7, 96);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave104(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(5, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(6, 98);
        break;
    case 2:
        call_onekernel_seq2seq_enc(7, 97);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave105(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(6, 99);
        break;
    case 1:
        call_onekernel_seq2seq_enc(7, 98);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave106(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_seq2seq_enc(7, 99);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave0(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave1(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave2(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave3(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave4(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave5(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave6(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave7(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave8(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave9(WaveInputParams *__restrict__ input,
                      WaveModelParams *__restrict__ model,
                      WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave10(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave11(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave12(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave13(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave14(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave15(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave16(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave17(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave18(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave19(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave20(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave21(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave22(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave23(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave24(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave25(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave26(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave27(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave28(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave29(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave30(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave31(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave32(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave33(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave34(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave35(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave36(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave37(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave38(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave39(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave40(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave41(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave42(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave43(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave44(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave45(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave46(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave47(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave48(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave49(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave50(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave51(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave52(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave53(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave54(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave55(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave56(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave57(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave58(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave59(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave60(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave61(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave62(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave63(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave64(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave65(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave66(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave67(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave68(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave69(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave70(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave71(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave72(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave73(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave74(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave75(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave76(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave77(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave78(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave79(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave80(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave81(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave82(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave83(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave84(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave85(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave86(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave87(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave88(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave89(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave90(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave91(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave92(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave93(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave94(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave95(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave96(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave97(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave98(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave99(WaveInputParams *__restrict__ input,
                       WaveModelParams *__restrict__ model,
                       WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave100(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave101(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave102(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave103(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave104(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave105(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave106(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave107(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave108(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave109(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave110(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave111(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave112(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave113(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave114(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave115(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave116(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(0, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave117(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(1, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave118(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(2, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave119(WaveInputParams *__restrict__ input,
                        WaveModelParams *__restrict__ model,
                        WaveOutputParams *__restrict__ output) {
    call_onekernel_seq2seq_dec(3, 29);
}

__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_0(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_1(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 1);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_2(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 2);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 1);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_3(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 3);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 2);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 1);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_4(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 4);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 3);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 2);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 1);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_5(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 5);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 4);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 3);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 2);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 1);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_6(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 6);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 5);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 4);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 3);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 2);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 1);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_7(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 7);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 6);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 5);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 4);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 3);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 2);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 1);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_8(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 8);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 7);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 6);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 5);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 4);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 3);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 2);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 1);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_9(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 9);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 8);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 7);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 6);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 5);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 4);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 3);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 2);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_10(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 10);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 9);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 8);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 7);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 6);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 5);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 4);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 3);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_11(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 11);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 10);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 9);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 8);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 7);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 6);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 5);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 4);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_12(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 12);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 11);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 10);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 9);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 8);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 7);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 6);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 5);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_13(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 13);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 12);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 11);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 10);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 9);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 8);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 7);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 6);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_14(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 14);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 13);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 12);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 11);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 10);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 9);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 8);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 7);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_15(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 15);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 14);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 13);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 12);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 11);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 10);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 9);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 8);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_16(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 16);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 15);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 14);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 13);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 12);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 11);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 10);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 9);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_17(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 17);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 16);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 15);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 14);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 13);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 12);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 11);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 10);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_18(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 18);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 17);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 16);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 15);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 14);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 13);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 12);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 11);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_19(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 19);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 18);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 17);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 16);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 15);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 14);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 13);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 12);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_20(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 20);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 19);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 18);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 17);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 16);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 15);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 14);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 13);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_21(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 21);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 20);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 19);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 18);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 17);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 16);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 15);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 14);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_22(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 22);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 21);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 20);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 19);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 18);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 17);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 16);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 15);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_23(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 23);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 22);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 21);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 20);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 19);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 18);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 17);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 16);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_24(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 24);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 23);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 22);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 21);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 20);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 19);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 18);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 17);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_25(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 25);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 24);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 23);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 22);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 21);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 20);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 19);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 18);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_26(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 26);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 25);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 24);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 23);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 22);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 21);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 20);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 19);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_27(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 27);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 26);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 25);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 24);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 23);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 22);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 21);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 20);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_28(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 28);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 27);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 26);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 25);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 24);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 23);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 22);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 21);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_29(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 29);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 28);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 27);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 26);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 25);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 24);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 23);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 22);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_30(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 30);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 29);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 28);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 27);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 26);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 25);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 24);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 23);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_31(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 31);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 30);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 29);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 28);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 27);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 26);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 25);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 24);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_32(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 32);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 31);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 30);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 29);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 28);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 27);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 26);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 25);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_33(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 33);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 32);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 31);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 30);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 29);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 28);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 27);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 26);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_34(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 34);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 33);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 32);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 31);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 30);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 29);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 28);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 27);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_35(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 35);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 34);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 33);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 32);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 31);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 30);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 29);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 28);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_36(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 36);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 35);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 34);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 33);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 32);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 31);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 30);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 29);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_37(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 37);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 36);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 35);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 34);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 33);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 32);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 31);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 30);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_38(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 38);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 37);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 36);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 35);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 34);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 33);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 32);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 31);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_39(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 39);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 38);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 37);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 36);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 35);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 34);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 33);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 32);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_40(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 40);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 39);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 38);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 37);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 36);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 35);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 34);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 33);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_41(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 41);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 40);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 39);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 38);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 37);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 36);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 35);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 34);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_42(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 42);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 41);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 40);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 39);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 38);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 37);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 36);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 35);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_43(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 43);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 42);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 41);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 40);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 39);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 38);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 37);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 36);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_44(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 44);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 43);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 42);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 41);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 40);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 39);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 38);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 37);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_45(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 45);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 44);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 43);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 42);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 41);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 40);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 39);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 38);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_46(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 46);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 45);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 44);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 43);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 42);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 41);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 40);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 39);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_47(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 47);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 46);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 45);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 44);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 43);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 42);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 41);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 40);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_48(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 48);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 47);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 46);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 45);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 44);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 43);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 42);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 41);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_49(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 49);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 48);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 47);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 46);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 45);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 44);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 43);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 42);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_50(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 50);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 49);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 48);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 47);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 46);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 45);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 44);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 43);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_51(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 51);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 50);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 49);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 48);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 47);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 46);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 45);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 44);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_52(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 52);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 51);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 50);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 49);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 48);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 47);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 46);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 45);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_53(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 53);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 52);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 51);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 50);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 49);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 48);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 47);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 46);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_54(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 54);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 53);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 52);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 51);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 50);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 49);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 48);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 47);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_55(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 55);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 54);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 53);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 52);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 51);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 50);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 49);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 48);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_56(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 56);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 55);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 54);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 53);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 52);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 51);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 50);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 49);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_57(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 57);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 56);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 55);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 54);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 53);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 52);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 51);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 50);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_58(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 58);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 57);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 56);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 55);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 54);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 53);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 52);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 51);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_59(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 59);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 58);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 57);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 56);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 55);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 54);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 53);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 52);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_60(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 60);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 59);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 58);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 57);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 56);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 55);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 54);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 53);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_61(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 61);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 60);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 59);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 58);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 57);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 56);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 55);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 54);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_62(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 62);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 61);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 60);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 59);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 58);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 57);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 56);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 55);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_63(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 63);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 62);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 61);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 60);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 59);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 58);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 57);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 56);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_64(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 64);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 63);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 62);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 61);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 60);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 59);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 58);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 57);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_65(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 65);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 64);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 63);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 62);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 61);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 60);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 59);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 58);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_66(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 66);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 65);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 64);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 63);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 62);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 61);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 60);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 59);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_67(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 67);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 66);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 65);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 64);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 63);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 62);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 61);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 60);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_68(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 68);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 67);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 66);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 65);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 64);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 63);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 62);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 61);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_69(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 69);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 68);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 67);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 66);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 65);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 64);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 63);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 62);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_70(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 70);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 69);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 68);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 67);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 66);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 65);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 64);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 63);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_71(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 71);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 70);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 69);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 68);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 67);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 66);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 65);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 64);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_72(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 72);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 71);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 70);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 69);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 68);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 67);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 66);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 65);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_73(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 73);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 72);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 71);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 70);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 69);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 68);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 67);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 66);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_74(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 74);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 73);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 72);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 71);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 70);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 69);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 68);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 67);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_75(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 75);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 74);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 73);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 72);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 71);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 70);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 69);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 68);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_76(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 76);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 75);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 74);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 73);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 72);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 71);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 70);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 69);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_77(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 77);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 76);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 75);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 74);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 73);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 72);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 71);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 70);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_78(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 78);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 77);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 76);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 75);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 74);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 73);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 72);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 71);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_79(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 79);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 78);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 77);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 76);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 75);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 74);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 73);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 72);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_80(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 80);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 79);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 78);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 77);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 76);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 75);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 74);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 73);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_81(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 81);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 80);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 79);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 78);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 77);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 76);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 75);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 74);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_82(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 82);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 81);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 80);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 79);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 78);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 77);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 76);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 75);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_83(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 83);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 82);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 81);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 80);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 79);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 78);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 77);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 76);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_84(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 84);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 83);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 82);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 81);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 80);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 79);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 78);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 77);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_85(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 85);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 84);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 83);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 82);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 81);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 80);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 79);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 78);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_86(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 86);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 85);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 84);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 83);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 82);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 81);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 80);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 79);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_87(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 87);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 86);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 85);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 84);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 83);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 82);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 81);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 80);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_88(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 88);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 87);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 86);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 85);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 84);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 83);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 82);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 81);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_89(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 89);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 88);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 87);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 86);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 85);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 84);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 83);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 82);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_90(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 90);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 89);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 88);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 87);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 86);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 85);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 84);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 83);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_91(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 91);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 90);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 89);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 88);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 87);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 86);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 85);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 84);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_92(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 92);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 91);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 90);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 89);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 88);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 87);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 86);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 85);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_93(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 93);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 92);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 91);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 90);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 89);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 88);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 87);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 86);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_94(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 94);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 93);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 92);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 91);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 90);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 89);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 88);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 87);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_95(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 95);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 94);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 93);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 92);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 91);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 90);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 89);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 88);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_96(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 96);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 95);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 94);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 93);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 92);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 91);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 90);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 89);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_97(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 97);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 96);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 95);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 94);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 93);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 92);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 91);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 90);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_98(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 98);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 97);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 96);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 95);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 94);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 93);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 92);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 91);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_99(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(0, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(1, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(2, 97);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(3, 96);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(4, 95);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(5, 94);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(6, 93);
        break;
    case 7:
        call_onekernel_compute_seq2seq_enc(7, 92);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_100(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(1, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(2, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(3, 97);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(4, 96);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(5, 95);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(6, 94);
        break;
    case 6:
        call_onekernel_compute_seq2seq_enc(7, 93);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_101(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(2, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(3, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(4, 97);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(5, 96);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(6, 95);
        break;
    case 5:
        call_onekernel_compute_seq2seq_enc(7, 94);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_102(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(3, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(4, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(5, 97);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(6, 96);
        break;
    case 4:
        call_onekernel_compute_seq2seq_enc(7, 95);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_103(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(4, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(5, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(6, 97);
        break;
    case 3:
        call_onekernel_compute_seq2seq_enc(7, 96);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_104(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(5, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(6, 98);
        break;
    case 2:
        call_onekernel_compute_seq2seq_enc(7, 97);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_105(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(6, 99);
        break;
    case 1:
        call_onekernel_compute_seq2seq_enc(7, 98);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_compute_106(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_seq2seq_enc(7, 99);
        break;
    }
}

__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_0(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_1(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_2(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_3(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_4(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_5(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_6(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_7(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_8(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_9(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_10(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_11(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_12(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_13(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_14(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_15(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_16(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_17(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_18(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_19(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_20(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_21(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_22(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_23(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_24(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_25(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_26(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_27(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_28(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_29(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_30(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_31(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_32(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_33(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_34(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_35(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_36(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_37(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_38(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_39(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_40(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_41(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_42(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_43(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_44(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_45(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_46(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_47(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_48(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_49(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_50(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_51(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_52(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_53(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_54(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_55(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_56(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_57(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_58(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_59(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_60(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_61(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_62(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_63(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_64(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_65(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_66(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_67(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_68(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_69(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_70(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_71(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_72(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_73(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_74(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_75(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_76(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_77(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_78(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_79(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_80(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_81(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_82(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_83(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_84(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_85(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_86(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_87(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_88(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_89(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_90(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_91(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_92(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_93(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_94(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_95(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_96(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_97(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_98(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_99(WaveInputParams *__restrict__ input,
                                WaveModelParams *__restrict__ model,
                                WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_100(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_101(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_102(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_103(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_104(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_105(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_106(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_107(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_108(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_109(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_110(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_111(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_112(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_113(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_114(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_115(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_116(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(0, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_117(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(1, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_118(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(2, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_compute_119(WaveInputParams *__restrict__ input,
                                 WaveModelParams *__restrict__ model,
                                 WaveOutputParams *__restrict__ output) {
    call_onekernel_compute_seq2seq_dec(3, 29);
}

__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_0(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_1(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 1);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_2(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 2);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 1);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_3(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 3);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 2);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 1);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_4(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 4);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 3);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 2);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 1);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_5(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 5);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 4);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 3);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 2);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 1);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_6(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 6);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 5);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 4);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 3);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 2);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 1);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_7(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 7);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 6);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 5);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 4);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 3);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 2);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 1);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_8(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 8);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 7);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 6);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 5);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 4);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 3);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 2);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 1);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_9(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 9);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 8);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 7);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 6);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 5);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 4);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 3);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 2);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_10(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 10);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 9);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 8);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 7);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 6);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 5);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 4);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 3);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_11(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 11);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 10);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 9);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 8);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 7);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 6);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 5);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 4);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_12(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 12);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 11);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 10);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 9);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 8);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 7);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 6);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 5);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_13(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 13);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 12);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 11);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 10);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 9);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 8);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 7);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 6);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_14(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 14);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 13);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 12);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 11);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 10);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 9);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 8);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 7);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_15(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 15);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 14);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 13);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 12);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 11);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 10);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 9);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 8);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_16(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 16);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 15);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 14);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 13);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 12);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 11);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 10);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 9);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_17(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 17);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 16);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 15);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 14);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 13);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 12);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 11);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 10);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_18(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 18);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 17);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 16);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 15);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 14);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 13);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 12);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 11);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_19(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 19);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 18);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 17);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 16);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 15);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 14);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 13);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 12);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_20(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 20);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 19);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 18);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 17);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 16);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 15);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 14);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 13);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_21(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 21);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 20);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 19);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 18);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 17);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 16);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 15);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 14);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_22(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 22);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 21);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 20);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 19);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 18);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 17);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 16);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 15);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_23(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 23);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 22);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 21);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 20);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 19);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 18);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 17);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 16);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_24(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 24);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 23);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 22);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 21);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 20);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 19);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 18);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 17);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_25(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 25);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 24);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 23);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 22);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 21);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 20);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 19);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 18);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_26(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 26);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 25);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 24);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 23);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 22);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 21);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 20);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 19);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_27(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 27);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 26);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 25);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 24);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 23);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 22);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 21);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 20);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_28(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 28);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 27);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 26);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 25);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 24);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 23);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 22);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 21);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_29(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 29);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 28);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 27);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 26);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 25);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 24);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 23);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 22);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_30(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 30);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 29);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 28);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 27);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 26);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 25);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 24);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 23);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_31(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 31);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 30);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 29);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 28);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 27);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 26);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 25);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 24);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_32(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 32);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 31);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 30);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 29);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 28);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 27);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 26);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 25);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_33(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 33);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 32);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 31);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 30);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 29);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 28);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 27);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 26);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_34(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 34);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 33);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 32);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 31);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 30);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 29);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 28);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 27);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_35(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 35);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 34);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 33);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 32);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 31);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 30);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 29);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 28);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_36(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 36);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 35);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 34);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 33);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 32);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 31);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 30);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 29);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_37(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 37);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 36);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 35);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 34);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 33);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 32);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 31);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 30);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_38(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 38);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 37);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 36);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 35);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 34);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 33);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 32);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 31);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_39(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 39);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 38);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 37);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 36);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 35);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 34);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 33);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 32);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_40(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 40);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 39);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 38);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 37);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 36);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 35);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 34);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 33);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_41(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 41);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 40);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 39);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 38);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 37);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 36);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 35);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 34);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_42(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 42);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 41);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 40);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 39);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 38);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 37);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 36);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 35);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_43(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 43);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 42);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 41);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 40);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 39);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 38);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 37);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 36);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_44(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 44);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 43);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 42);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 41);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 40);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 39);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 38);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 37);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_45(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 45);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 44);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 43);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 42);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 41);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 40);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 39);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 38);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_46(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 46);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 45);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 44);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 43);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 42);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 41);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 40);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 39);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_47(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 47);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 46);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 45);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 44);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 43);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 42);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 41);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 40);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_48(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 48);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 47);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 46);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 45);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 44);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 43);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 42);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 41);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_49(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 49);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 48);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 47);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 46);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 45);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 44);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 43);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 42);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_50(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 50);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 49);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 48);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 47);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 46);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 45);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 44);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 43);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_51(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 51);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 50);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 49);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 48);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 47);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 46);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 45);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 44);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_52(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 52);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 51);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 50);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 49);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 48);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 47);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 46);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 45);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_53(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 53);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 52);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 51);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 50);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 49);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 48);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 47);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 46);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_54(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 54);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 53);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 52);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 51);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 50);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 49);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 48);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 47);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_55(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 55);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 54);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 53);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 52);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 51);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 50);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 49);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 48);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_56(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 56);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 55);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 54);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 53);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 52);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 51);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 50);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 49);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_57(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 57);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 56);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 55);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 54);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 53);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 52);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 51);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 50);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_58(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 58);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 57);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 56);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 55);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 54);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 53);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 52);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 51);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_59(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 59);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 58);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 57);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 56);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 55);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 54);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 53);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 52);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_60(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 60);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 59);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 58);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 57);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 56);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 55);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 54);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 53);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_61(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 61);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 60);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 59);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 58);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 57);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 56);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 55);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 54);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_62(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 62);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 61);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 60);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 59);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 58);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 57);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 56);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 55);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_63(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 63);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 62);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 61);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 60);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 59);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 58);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 57);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 56);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_64(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 64);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 63);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 62);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 61);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 60);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 59);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 58);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 57);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_65(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 65);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 64);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 63);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 62);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 61);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 60);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 59);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 58);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_66(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 66);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 65);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 64);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 63);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 62);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 61);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 60);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 59);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_67(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 67);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 66);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 65);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 64);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 63);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 62);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 61);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 60);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_68(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 68);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 67);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 66);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 65);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 64);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 63);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 62);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 61);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_69(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 69);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 68);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 67);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 66);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 65);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 64);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 63);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 62);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_70(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 70);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 69);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 68);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 67);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 66);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 65);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 64);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 63);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_71(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 71);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 70);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 69);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 68);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 67);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 66);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 65);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 64);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_72(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 72);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 71);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 70);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 69);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 68);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 67);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 66);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 65);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_73(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 73);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 72);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 71);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 70);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 69);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 68);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 67);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 66);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_74(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 74);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 73);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 72);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 71);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 70);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 69);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 68);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 67);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_75(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 75);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 74);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 73);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 72);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 71);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 70);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 69);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 68);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_76(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 76);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 75);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 74);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 73);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 72);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 71);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 70);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 69);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_77(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 77);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 76);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 75);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 74);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 73);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 72);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 71);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 70);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_78(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 78);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 77);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 76);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 75);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 74);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 73);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 72);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 71);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_79(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 79);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 78);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 77);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 76);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 75);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 74);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 73);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 72);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_80(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 80);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 79);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 78);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 77);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 76);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 75);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 74);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 73);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_81(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 81);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 80);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 79);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 78);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 77);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 76);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 75);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 74);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_82(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 82);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 81);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 80);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 79);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 78);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 77);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 76);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 75);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_83(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 83);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 82);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 81);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 80);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 79);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 78);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 77);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 76);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_84(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 84);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 83);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 82);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 81);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 80);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 79);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 78);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 77);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_85(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 85);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 84);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 83);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 82);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 81);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 80);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 79);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 78);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_86(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 86);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 85);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 84);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 83);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 82);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 81);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 80);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 79);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_87(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 87);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 86);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 85);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 84);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 83);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 82);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 81);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 80);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_88(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 88);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 87);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 86);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 85);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 84);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 83);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 82);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 81);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_89(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 89);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 88);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 87);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 86);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 85);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 84);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 83);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 82);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_90(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 90);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 89);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 88);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 87);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 86);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 85);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 84);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 83);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_91(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 91);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 90);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 89);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 88);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 87);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 86);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 85);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 84);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_92(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 92);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 91);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 90);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 89);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 88);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 87);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 86);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 85);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_93(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 93);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 92);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 91);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 90);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 89);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 88);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 87);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 86);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_94(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 94);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 93);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 92);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 91);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 90);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 89);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 88);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 87);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_95(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 95);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 94);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 93);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 92);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 91);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 90);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 89);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 88);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_96(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 96);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 95);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 94);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 93);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 92);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 91);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 90);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 89);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_97(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 97);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 96);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 95);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 94);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 93);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 92);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 91);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 90);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_98(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 98);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 97);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 96);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 95);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 94);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 93);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 92);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 91);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_99(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(0, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(1, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(2, 97);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(3, 96);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(4, 95);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(5, 94);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(6, 93);
        break;
    case 7:
        call_onekernel_solve_seq2seq_enc(7, 92);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_100(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(1, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(2, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(3, 97);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(4, 96);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(5, 95);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(6, 94);
        break;
    case 6:
        call_onekernel_solve_seq2seq_enc(7, 93);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_101(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(2, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(3, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(4, 97);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(5, 96);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(6, 95);
        break;
    case 5:
        call_onekernel_solve_seq2seq_enc(7, 94);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_102(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(3, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(4, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(5, 97);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(6, 96);
        break;
    case 4:
        call_onekernel_solve_seq2seq_enc(7, 95);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_103(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(4, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(5, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(6, 97);
        break;
    case 3:
        call_onekernel_solve_seq2seq_enc(7, 96);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_104(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(5, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(6, 98);
        break;
    case 2:
        call_onekernel_solve_seq2seq_enc(7, 97);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_105(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(6, 99);
        break;
    case 1:
        call_onekernel_solve_seq2seq_enc(7, 98);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_enc_wave_solve_106(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_seq2seq_enc(7, 99);
        break;
    }
}

__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_0(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_1(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_2(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_3(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 0);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_4(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_5(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_6(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_7(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 1);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_8(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_9(WaveInputParams *__restrict__ input,
                             WaveModelParams *__restrict__ model,
                             WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_10(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_11(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 2);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_12(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_13(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_14(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_15(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 3);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_16(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_17(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_18(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_19(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 4);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_20(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_21(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_22(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_23(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 5);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_24(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_25(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_26(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_27(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 6);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_28(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_29(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_30(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_31(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 7);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_32(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_33(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_34(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_35(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 8);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_36(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_37(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_38(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_39(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 9);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_40(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_41(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_42(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_43(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 10);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_44(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_45(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_46(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_47(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 11);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_48(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_49(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_50(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_51(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 12);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_52(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_53(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_54(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_55(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 13);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_56(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_57(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_58(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_59(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 14);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_60(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_61(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_62(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_63(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 15);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_64(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_65(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_66(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_67(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 16);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_68(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_69(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_70(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_71(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 17);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_72(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_73(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_74(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_75(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 18);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_76(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_77(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_78(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_79(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 19);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_80(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_81(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_82(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_83(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 20);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_84(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_85(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_86(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_87(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 21);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_88(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_89(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_90(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_91(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 22);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_92(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_93(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_94(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_95(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 23);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_96(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_97(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_98(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_99(WaveInputParams *__restrict__ input,
                              WaveModelParams *__restrict__ model,
                              WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 24);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_100(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_101(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_102(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_103(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 25);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_104(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_105(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_106(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_107(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 26);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_108(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_109(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_110(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_111(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 27);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_112(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_113(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_114(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_115(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 28);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_116(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(0, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_117(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(1, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_118(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(2, 29);
}
__global__ void __launch_bounds__(256, 1)
    seq2seq_dec_wave_solve_119(WaveInputParams *__restrict__ input,
                               WaveModelParams *__restrict__ model,
                               WaveOutputParams *__restrict__ output) {
    call_onekernel_solve_seq2seq_dec(3, 29);
}
