#include "net/lstm_cell_net_100_10_bs4/WavefrontParamsBS4.h"
#define COLUMNS_PER_BLOCK 32 // one block compute 32 colums
#define THREAD_NUMS_PER_BLOCK 256
#define HIDDENSIZE 256
#define INPUTSIZE HIDDENSIZE
#define CELL_NUM 10
#define NUM_LAYER 100
#define call_onekernel(cell, step)                                             \
    {                                                                          \
        onekernel_fuse_opt_v2(blockIdx.x & 0x7,                                \
                              input + step * CELL_NUM + cell, model + cell,    \
                              output + step * CELL_NUM + cell);                \
    }

__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__device__ void onekernel_fuse(dim3 blockIdx1, const float *input_i,
                               const float4 *weight_w, float4 *wi,
                               const float *input_h, const float4 *weight_u,
                               float4 *uh, float4 *bias, float *state_c,
                               float *state_h) {

    if (blockIdx1.x < 8) {
        __shared__ float4 nndense_output[COLUMNS_PER_BLOCK];
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        nndense_output[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
        float temp[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
        const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE;
        for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
            const float data = input_i[vectorRow];
            float4 res = weight_w[kStart];
            temp[0] = fma(res.x, data, temp[0]);
            temp[1] = fma(res.y, data, temp[1]);
            temp[2] = fma(res.z, data, temp[2]);
            temp[3] = fma(res.w, data, temp[3]);
        }
        __syncthreads();

        atomicAdd(&nndense_output[lane_id].x, temp[0]);
        atomicAdd(&nndense_output[lane_id].y, temp[1]);
        atomicAdd(&nndense_output[lane_id].z, temp[2]);
        atomicAdd(&nndense_output[lane_id].w, temp[3]);
        __syncthreads();
        if (warp_id == 0) {
            wi[colOffset] = nndense_output[lane_id];
        }
    }
    /********************************************************************/
    if (blockIdx1.x < 8) {
        __shared__ float4 nndense_output[COLUMNS_PER_BLOCK];
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        nndense_output[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
        float temp[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
        const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
        int vectorRow = ROWS * warp_id;
        int kStart =
            vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        int kEnd = kStart + ROWS * HIDDENSIZE;
        for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
            const float data = input_h[vectorRow];
            float4 res = weight_u[kStart];
            temp[0] = fma(res.x, data, temp[0]);
            temp[1] = fma(res.y, data, temp[1]);
            temp[2] = fma(res.z, data, temp[2]);
            temp[3] = fma(res.w, data, temp[3]);
        }
        __syncthreads();

        atomicAdd(&nndense_output[lane_id].x, temp[0]);
        atomicAdd(&nndense_output[lane_id].y, temp[1]);
        atomicAdd(&nndense_output[lane_id].z, temp[2]);
        atomicAdd(&nndense_output[lane_id].w, temp[3]);
        __syncthreads();
        if (warp_id == 0) {
            uh[colOffset] = nndense_output[lane_id];
        }
    }
    /*********************************************************************/
    // global barrier
    __syncthreads();
    if (blockIdx1.x < 8) {
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 0x1f;
        const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
        if (warp_id == 0) {
            float x, y, z, w;
            x = wi[colOffset].x + uh[colOffset].x + bias[colOffset].x;
            y = wi[colOffset].y + uh[colOffset].y + bias[colOffset].y;
            z = wi[colOffset].z + uh[colOffset].z + bias[colOffset].z;
            w = wi[colOffset].w + uh[colOffset].w + bias[colOffset].w;

            x = sigmoid(x);
            y = tanh(y);
            w = sigmoid(w);
            z = sigmoid(z + 1.0000f) * state_c[colOffset];
            state_c[colOffset] = fma(x, y, z);
            state_h[colOffset] = (tanh(state_c[colOffset])) * w;
        }
    }
}

__device__ void onekernel_fuse_opt_v1(dim3 blockIdx1, const float *input_i,
                                      const float4 *weight_w, float4 *wi,
                                      const float *input_h,
                                      const float4 *weight_u, float4 *uh,
                                      float4 *bias, float *state_c,
                                      float *state_h) {

    __shared__ float4 nndense_output1[COLUMNS_PER_BLOCK];
    __shared__ float4 nndense_output2[COLUMNS_PER_BLOCK];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    nndense_output1[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    nndense_output2[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    float temp1[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    float temp2[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        const float data = input_i[vectorRow];
        float4 res = weight_w[kStart];
        temp1[0] = fma(res.x, data, temp1[0]);
        temp1[1] = fma(res.y, data, temp1[1]);
        temp1[2] = fma(res.z, data, temp1[2]);
        temp1[3] = fma(res.w, data, temp1[3]);
        const float data2 = input_h[vectorRow];
        float4 res2 = weight_u[kStart];
        temp2[0] = fma(res2.x, data2, temp2[0]);
        temp2[1] = fma(res2.y, data2, temp2[1]);
        temp2[2] = fma(res2.z, data2, temp2[2]);
        temp2[3] = fma(res2.w, data2, temp2[3]);
    }
    __syncthreads();

    atomicAdd(&nndense_output1[lane_id].x, temp1[0]);
    atomicAdd(&nndense_output1[lane_id].y, temp1[1]);
    atomicAdd(&nndense_output1[lane_id].z, temp1[2]);
    atomicAdd(&nndense_output1[lane_id].w, temp1[3]);
    atomicAdd(&nndense_output2[lane_id].x, temp2[0]);
    atomicAdd(&nndense_output2[lane_id].y, temp2[1]);
    atomicAdd(&nndense_output2[lane_id].z, temp2[2]);
    atomicAdd(&nndense_output2[lane_id].w, temp2[3]);
    __syncthreads();
    if (warp_id == 0) {
        wi[colOffset] = nndense_output1[lane_id];
        uh[colOffset] = nndense_output2[lane_id];
        float x, y, z, w;
        x = wi[colOffset].x + uh[colOffset].x + bias[colOffset].x;
        y = wi[colOffset].y + uh[colOffset].y + bias[colOffset].y;
        z = wi[colOffset].z + uh[colOffset].z + bias[colOffset].z;
        w = wi[colOffset].w + uh[colOffset].w + bias[colOffset].w;

        x = sigmoid(x);
        y = tanh(y);
        w = sigmoid(w);
        z = sigmoid(z + 1.0000f) * state_c[colOffset];
        state_c[colOffset] = fma(x, y, z);
        state_h[colOffset] = (tanh(state_c[colOffset])) * w;
    }
}

__device__ void
onekernel_fuse_opt_v2(dim3 blockIdx1, WaveInputParamsBS4 *__restrict__ input,
                      WaveModelParamsBS4 *__restrict__ model,
                      WaveOutputParamsBS4 *__restrict__ output) {

    __shared__ float4 nndense_output1[4][COLUMNS_PER_BLOCK];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    for (int i = 0; i < 4; ++i)
        nndense_output1[i][lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};

    float4 temp1[4] = {0.0000f};

    const int ROWS = INPUTSIZE / (THREAD_NUMS_PER_BLOCK / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * HIDDENSIZE + blockIdx1.x * COLUMNS_PER_BLOCK + lane_id;
    int kEnd = kStart + ROWS * HIDDENSIZE;
    for (; kStart < kEnd; kStart += HIDDENSIZE, ++vectorRow) {
        const float4 data = input->input_i[vectorRow];
        float4 res = model->weight_w[kStart];
        const float4 data2 = input->input_h[vectorRow];
        float4 res2 = model->weight_u[kStart];
        temp1[0].x = fma(res.x, data.x, temp1[0].x);
        temp1[0].y = fma(res.y, data.x, temp1[0].y);
        temp1[0].z = fma(res.z, data.x, temp1[0].z);
        temp1[0].w = fma(res.w, data.x, temp1[0].w);

        temp1[0].x = fma(res2.x, data2.x, temp1[0].x);
        temp1[0].y = fma(res2.y, data2.x, temp1[0].y);
        temp1[0].z = fma(res2.z, data2.x, temp1[0].z);
        temp1[0].w = fma(res2.w, data2.x, temp1[0].w);
        // batch 1
        temp1[1].x = fma(res.x, data.y, temp1[1].x);
        temp1[1].y = fma(res.y, data.y, temp1[1].y);
        temp1[1].z = fma(res.z, data.y, temp1[1].z);
        temp1[1].w = fma(res.w, data.y, temp1[1].w);
        temp1[1].x = fma(res2.x, data2.y, temp1[1].x);
        temp1[1].y = fma(res2.y, data2.y, temp1[1].y);
        temp1[1].z = fma(res2.z, data2.y, temp1[1].z);
        temp1[1].w = fma(res2.w, data2.y, temp1[1].w);
        // batch 2
        temp1[2].x = fma(res.x, data.z, temp1[2].x);
        temp1[2].y = fma(res.y, data.z, temp1[2].y);
        temp1[2].z = fma(res.z, data.z, temp1[2].z);
        temp1[2].w = fma(res.w, data.z, temp1[2].w);
        temp1[2].x = fma(res2.x, data2.z, temp1[2].x);
        temp1[2].y = fma(res2.y, data2.z, temp1[2].y);
        temp1[2].z = fma(res2.z, data2.z, temp1[2].z);
        temp1[2].w = fma(res2.w, data2.z, temp1[2].w);
        // batch3
        temp1[3].x = fma(res.x, data.w, temp1[3].x);
        temp1[3].y = fma(res.y, data.w, temp1[3].y);
        temp1[3].z = fma(res.z, data.w, temp1[3].z);
        temp1[3].w = fma(res.w, data.w, temp1[3].w);
        temp1[3].x = fma(res2.x, data2.w, temp1[3].x);
        temp1[3].y = fma(res2.y, data2.w, temp1[3].y);
        temp1[3].z = fma(res2.z, data2.w, temp1[3].z);
        temp1[3].w = fma(res2.w, data2.w, temp1[3].w);
    }
    __syncthreads();
    for (int batchIndex = 0; batchIndex < 4; ++batchIndex) {
        atomicAdd(&nndense_output1[batchIndex][lane_id].x, temp1[batchIndex].x);
        atomicAdd(&nndense_output1[batchIndex][lane_id].y, temp1[batchIndex].y);
        atomicAdd(&nndense_output1[batchIndex][lane_id].z, temp1[batchIndex].z);
        atomicAdd(&nndense_output1[batchIndex][lane_id].w, temp1[batchIndex].w);
    }
    __syncthreads();
    if (warp_id == 0) {
        float4 bs0, bs1, bs2, bs3, state_h;
        float4 state_c = output->state_c[colOffset];
        float4 bias_t = model->bias[colOffset];
        bs0.x = nndense_output1[0][lane_id].x + bias_t.x;
        bs0.y = nndense_output1[0][lane_id].y + bias_t.y;
        bs0.z = nndense_output1[0][lane_id].z + bias_t.z;
        bs0.w = nndense_output1[0][lane_id].w + bias_t.w;
        bs0.x = sigmoid(bs0.x);
        bs0.y = tanh(bs0.y);
        bs0.w = sigmoid(bs0.w);
        bs0.z = sigmoid(bs0.z + 1.0000f) * state_c.x;
        state_c.x = fma(bs0.x, bs0.y, bs0.z);
        // batch 1
        bs1.x = nndense_output1[1][lane_id].x + bias_t.x;
        bs1.y = nndense_output1[1][lane_id].y + bias_t.y;
        bs1.z = nndense_output1[1][lane_id].z + bias_t.z;
        bs1.w = nndense_output1[1][lane_id].w + bias_t.w;
        bs1.x = sigmoid(bs1.x);
        bs1.y = tanh(bs1.y);
        bs1.w = sigmoid(bs1.w);
        bs1.z = sigmoid(bs1.z + 1.0000f) * state_c.y;
        state_c.y = fma(bs1.x, bs1.y, bs1.z);
        // batch 2
        bs2.x = nndense_output1[2][lane_id].x + bias_t.x;
        bs2.y = nndense_output1[2][lane_id].y + bias_t.y;
        bs2.z = nndense_output1[2][lane_id].z + bias_t.z;
        bs2.w = nndense_output1[2][lane_id].w + bias_t.w;
        bs2.x = sigmoid(bs2.x);
        bs2.y = tanh(bs2.y);
        bs2.w = sigmoid(bs2.w);
        bs2.z = sigmoid(bs2.z + 1.0000f) * state_c.z;
        state_c.z = fma(bs2.x, bs2.y, bs2.z);
        // batch 3
        bs3.x = nndense_output1[3][lane_id].x + bias_t.x;
        bs3.y = nndense_output1[3][lane_id].y + bias_t.y;
        bs3.z = nndense_output1[3][lane_id].z + bias_t.z;
        bs3.w = nndense_output1[3][lane_id].w + bias_t.w;
        bs3.x = sigmoid(bs3.x);
        bs3.y = tanh(bs3.y);
        bs3.w = sigmoid(bs3.w);
        bs3.z = sigmoid(bs3.z + 1.0000f) * state_c.w;
        state_c.w = fma(bs3.x, bs3.y, bs3.z);
        //
        state_h.x = (tanh(state_c.x)) * bs0.w;
        state_h.y = (tanh(state_c.y)) * bs1.w;
        state_h.z = (tanh(state_c.z)) * bs2.w;
        state_h.w = (tanh(state_c.w)) * bs3.w;
        output->state_c[colOffset] = state_c;
        output->state_h[colOffset] = state_h;
    }
}

__global__ void __launch_bounds__(256, 1)
    wave0(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave1(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 1);
        break;
    case 1:
        call_onekernel(1, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave2(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 2);
        break;
    case 1:
        call_onekernel(1, 1);
        break;
    case 2:
        call_onekernel(2, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave3(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 3);
        break;
    case 1:
        call_onekernel(1, 2);
        break;
    case 2:
        call_onekernel(2, 1);
        break;
    case 3:
        call_onekernel(3, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave4(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 4);
        break;
    case 1:
        call_onekernel(1, 3);
        break;
    case 2:
        call_onekernel(2, 2);
        break;
    case 3:
        call_onekernel(3, 1);
        break;
    case 4:
        call_onekernel(4, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave5(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 5);
        break;
    case 1:
        call_onekernel(1, 4);
        break;
    case 2:
        call_onekernel(2, 3);
        break;
    case 3:
        call_onekernel(3, 2);
        break;
    case 4:
        call_onekernel(4, 1);
        break;
    case 5:
        call_onekernel(5, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave6(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 6);
        break;
    case 1:
        call_onekernel(1, 5);
        break;
    case 2:
        call_onekernel(2, 4);
        break;
    case 3:
        call_onekernel(3, 3);
        break;
    case 4:
        call_onekernel(4, 2);
        break;
    case 5:
        call_onekernel(5, 1);
        break;
    case 6:
        call_onekernel(6, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave7(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 7);
        break;
    case 1:
        call_onekernel(1, 6);
        break;
    case 2:
        call_onekernel(2, 5);
        break;
    case 3:
        call_onekernel(3, 4);
        break;
    case 4:
        call_onekernel(4, 3);
        break;
    case 5:
        call_onekernel(5, 2);
        break;
    case 6:
        call_onekernel(6, 1);
        break;
    case 7:
        call_onekernel(7, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave8(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 8);
        break;
    case 1:
        call_onekernel(1, 7);
        break;
    case 2:
        call_onekernel(2, 6);
        break;
    case 3:
        call_onekernel(3, 5);
        break;
    case 4:
        call_onekernel(4, 4);
        break;
    case 5:
        call_onekernel(5, 3);
        break;
    case 6:
        call_onekernel(6, 2);
        break;
    case 7:
        call_onekernel(7, 1);
        break;
    case 8:
        call_onekernel(8, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave9(WaveInputParamsBS4 *__restrict__ input,
          WaveModelParamsBS4 *__restrict__ model,
          WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 9);
        break;
    case 1:
        call_onekernel(1, 8);
        break;
    case 2:
        call_onekernel(2, 7);
        break;
    case 3:
        call_onekernel(3, 6);
        break;
    case 4:
        call_onekernel(4, 5);
        break;
    case 5:
        call_onekernel(5, 4);
        break;
    case 6:
        call_onekernel(6, 3);
        break;
    case 7:
        call_onekernel(7, 2);
        break;
    case 8:
        call_onekernel(8, 1);
        break;
    case 9:
        call_onekernel(9, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave10(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 10);
        break;
    case 1:
        call_onekernel(1, 9);
        break;
    case 2:
        call_onekernel(2, 8);
        break;
    case 3:
        call_onekernel(3, 7);
        break;
    case 4:
        call_onekernel(4, 6);
        break;
    case 5:
        call_onekernel(5, 5);
        break;
    case 6:
        call_onekernel(6, 4);
        break;
    case 7:
        call_onekernel(7, 3);
        break;
    case 8:
        call_onekernel(8, 2);
        break;
    case 9:
        call_onekernel(9, 1);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave11(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 11);
        break;
    case 1:
        call_onekernel(1, 10);
        break;
    case 2:
        call_onekernel(2, 9);
        break;
    case 3:
        call_onekernel(3, 8);
        break;
    case 4:
        call_onekernel(4, 7);
        break;
    case 5:
        call_onekernel(5, 6);
        break;
    case 6:
        call_onekernel(6, 5);
        break;
    case 7:
        call_onekernel(7, 4);
        break;
    case 8:
        call_onekernel(8, 3);
        break;
    case 9:
        call_onekernel(9, 2);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave12(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 12);
        break;
    case 1:
        call_onekernel(1, 11);
        break;
    case 2:
        call_onekernel(2, 10);
        break;
    case 3:
        call_onekernel(3, 9);
        break;
    case 4:
        call_onekernel(4, 8);
        break;
    case 5:
        call_onekernel(5, 7);
        break;
    case 6:
        call_onekernel(6, 6);
        break;
    case 7:
        call_onekernel(7, 5);
        break;
    case 8:
        call_onekernel(8, 4);
        break;
    case 9:
        call_onekernel(9, 3);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave13(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 13);
        break;
    case 1:
        call_onekernel(1, 12);
        break;
    case 2:
        call_onekernel(2, 11);
        break;
    case 3:
        call_onekernel(3, 10);
        break;
    case 4:
        call_onekernel(4, 9);
        break;
    case 5:
        call_onekernel(5, 8);
        break;
    case 6:
        call_onekernel(6, 7);
        break;
    case 7:
        call_onekernel(7, 6);
        break;
    case 8:
        call_onekernel(8, 5);
        break;
    case 9:
        call_onekernel(9, 4);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave14(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 14);
        break;
    case 1:
        call_onekernel(1, 13);
        break;
    case 2:
        call_onekernel(2, 12);
        break;
    case 3:
        call_onekernel(3, 11);
        break;
    case 4:
        call_onekernel(4, 10);
        break;
    case 5:
        call_onekernel(5, 9);
        break;
    case 6:
        call_onekernel(6, 8);
        break;
    case 7:
        call_onekernel(7, 7);
        break;
    case 8:
        call_onekernel(8, 6);
        break;
    case 9:
        call_onekernel(9, 5);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave15(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 15);
        break;
    case 1:
        call_onekernel(1, 14);
        break;
    case 2:
        call_onekernel(2, 13);
        break;
    case 3:
        call_onekernel(3, 12);
        break;
    case 4:
        call_onekernel(4, 11);
        break;
    case 5:
        call_onekernel(5, 10);
        break;
    case 6:
        call_onekernel(6, 9);
        break;
    case 7:
        call_onekernel(7, 8);
        break;
    case 8:
        call_onekernel(8, 7);
        break;
    case 9:
        call_onekernel(9, 6);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave16(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 16);
        break;
    case 1:
        call_onekernel(1, 15);
        break;
    case 2:
        call_onekernel(2, 14);
        break;
    case 3:
        call_onekernel(3, 13);
        break;
    case 4:
        call_onekernel(4, 12);
        break;
    case 5:
        call_onekernel(5, 11);
        break;
    case 6:
        call_onekernel(6, 10);
        break;
    case 7:
        call_onekernel(7, 9);
        break;
    case 8:
        call_onekernel(8, 8);
        break;
    case 9:
        call_onekernel(9, 7);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave17(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 17);
        break;
    case 1:
        call_onekernel(1, 16);
        break;
    case 2:
        call_onekernel(2, 15);
        break;
    case 3:
        call_onekernel(3, 14);
        break;
    case 4:
        call_onekernel(4, 13);
        break;
    case 5:
        call_onekernel(5, 12);
        break;
    case 6:
        call_onekernel(6, 11);
        break;
    case 7:
        call_onekernel(7, 10);
        break;
    case 8:
        call_onekernel(8, 9);
        break;
    case 9:
        call_onekernel(9, 8);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave18(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 18);
        break;
    case 1:
        call_onekernel(1, 17);
        break;
    case 2:
        call_onekernel(2, 16);
        break;
    case 3:
        call_onekernel(3, 15);
        break;
    case 4:
        call_onekernel(4, 14);
        break;
    case 5:
        call_onekernel(5, 13);
        break;
    case 6:
        call_onekernel(6, 12);
        break;
    case 7:
        call_onekernel(7, 11);
        break;
    case 8:
        call_onekernel(8, 10);
        break;
    case 9:
        call_onekernel(9, 9);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave19(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 19);
        break;
    case 1:
        call_onekernel(1, 18);
        break;
    case 2:
        call_onekernel(2, 17);
        break;
    case 3:
        call_onekernel(3, 16);
        break;
    case 4:
        call_onekernel(4, 15);
        break;
    case 5:
        call_onekernel(5, 14);
        break;
    case 6:
        call_onekernel(6, 13);
        break;
    case 7:
        call_onekernel(7, 12);
        break;
    case 8:
        call_onekernel(8, 11);
        break;
    case 9:
        call_onekernel(9, 10);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave20(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 20);
        break;
    case 1:
        call_onekernel(1, 19);
        break;
    case 2:
        call_onekernel(2, 18);
        break;
    case 3:
        call_onekernel(3, 17);
        break;
    case 4:
        call_onekernel(4, 16);
        break;
    case 5:
        call_onekernel(5, 15);
        break;
    case 6:
        call_onekernel(6, 14);
        break;
    case 7:
        call_onekernel(7, 13);
        break;
    case 8:
        call_onekernel(8, 12);
        break;
    case 9:
        call_onekernel(9, 11);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave21(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 21);
        break;
    case 1:
        call_onekernel(1, 20);
        break;
    case 2:
        call_onekernel(2, 19);
        break;
    case 3:
        call_onekernel(3, 18);
        break;
    case 4:
        call_onekernel(4, 17);
        break;
    case 5:
        call_onekernel(5, 16);
        break;
    case 6:
        call_onekernel(6, 15);
        break;
    case 7:
        call_onekernel(7, 14);
        break;
    case 8:
        call_onekernel(8, 13);
        break;
    case 9:
        call_onekernel(9, 12);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave22(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 22);
        break;
    case 1:
        call_onekernel(1, 21);
        break;
    case 2:
        call_onekernel(2, 20);
        break;
    case 3:
        call_onekernel(3, 19);
        break;
    case 4:
        call_onekernel(4, 18);
        break;
    case 5:
        call_onekernel(5, 17);
        break;
    case 6:
        call_onekernel(6, 16);
        break;
    case 7:
        call_onekernel(7, 15);
        break;
    case 8:
        call_onekernel(8, 14);
        break;
    case 9:
        call_onekernel(9, 13);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave23(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 23);
        break;
    case 1:
        call_onekernel(1, 22);
        break;
    case 2:
        call_onekernel(2, 21);
        break;
    case 3:
        call_onekernel(3, 20);
        break;
    case 4:
        call_onekernel(4, 19);
        break;
    case 5:
        call_onekernel(5, 18);
        break;
    case 6:
        call_onekernel(6, 17);
        break;
    case 7:
        call_onekernel(7, 16);
        break;
    case 8:
        call_onekernel(8, 15);
        break;
    case 9:
        call_onekernel(9, 14);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave24(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 24);
        break;
    case 1:
        call_onekernel(1, 23);
        break;
    case 2:
        call_onekernel(2, 22);
        break;
    case 3:
        call_onekernel(3, 21);
        break;
    case 4:
        call_onekernel(4, 20);
        break;
    case 5:
        call_onekernel(5, 19);
        break;
    case 6:
        call_onekernel(6, 18);
        break;
    case 7:
        call_onekernel(7, 17);
        break;
    case 8:
        call_onekernel(8, 16);
        break;
    case 9:
        call_onekernel(9, 15);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave25(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 25);
        break;
    case 1:
        call_onekernel(1, 24);
        break;
    case 2:
        call_onekernel(2, 23);
        break;
    case 3:
        call_onekernel(3, 22);
        break;
    case 4:
        call_onekernel(4, 21);
        break;
    case 5:
        call_onekernel(5, 20);
        break;
    case 6:
        call_onekernel(6, 19);
        break;
    case 7:
        call_onekernel(7, 18);
        break;
    case 8:
        call_onekernel(8, 17);
        break;
    case 9:
        call_onekernel(9, 16);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave26(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 26);
        break;
    case 1:
        call_onekernel(1, 25);
        break;
    case 2:
        call_onekernel(2, 24);
        break;
    case 3:
        call_onekernel(3, 23);
        break;
    case 4:
        call_onekernel(4, 22);
        break;
    case 5:
        call_onekernel(5, 21);
        break;
    case 6:
        call_onekernel(6, 20);
        break;
    case 7:
        call_onekernel(7, 19);
        break;
    case 8:
        call_onekernel(8, 18);
        break;
    case 9:
        call_onekernel(9, 17);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave27(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 27);
        break;
    case 1:
        call_onekernel(1, 26);
        break;
    case 2:
        call_onekernel(2, 25);
        break;
    case 3:
        call_onekernel(3, 24);
        break;
    case 4:
        call_onekernel(4, 23);
        break;
    case 5:
        call_onekernel(5, 22);
        break;
    case 6:
        call_onekernel(6, 21);
        break;
    case 7:
        call_onekernel(7, 20);
        break;
    case 8:
        call_onekernel(8, 19);
        break;
    case 9:
        call_onekernel(9, 18);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave28(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 28);
        break;
    case 1:
        call_onekernel(1, 27);
        break;
    case 2:
        call_onekernel(2, 26);
        break;
    case 3:
        call_onekernel(3, 25);
        break;
    case 4:
        call_onekernel(4, 24);
        break;
    case 5:
        call_onekernel(5, 23);
        break;
    case 6:
        call_onekernel(6, 22);
        break;
    case 7:
        call_onekernel(7, 21);
        break;
    case 8:
        call_onekernel(8, 20);
        break;
    case 9:
        call_onekernel(9, 19);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave29(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 29);
        break;
    case 1:
        call_onekernel(1, 28);
        break;
    case 2:
        call_onekernel(2, 27);
        break;
    case 3:
        call_onekernel(3, 26);
        break;
    case 4:
        call_onekernel(4, 25);
        break;
    case 5:
        call_onekernel(5, 24);
        break;
    case 6:
        call_onekernel(6, 23);
        break;
    case 7:
        call_onekernel(7, 22);
        break;
    case 8:
        call_onekernel(8, 21);
        break;
    case 9:
        call_onekernel(9, 20);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave30(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 30);
        break;
    case 1:
        call_onekernel(1, 29);
        break;
    case 2:
        call_onekernel(2, 28);
        break;
    case 3:
        call_onekernel(3, 27);
        break;
    case 4:
        call_onekernel(4, 26);
        break;
    case 5:
        call_onekernel(5, 25);
        break;
    case 6:
        call_onekernel(6, 24);
        break;
    case 7:
        call_onekernel(7, 23);
        break;
    case 8:
        call_onekernel(8, 22);
        break;
    case 9:
        call_onekernel(9, 21);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave31(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 31);
        break;
    case 1:
        call_onekernel(1, 30);
        break;
    case 2:
        call_onekernel(2, 29);
        break;
    case 3:
        call_onekernel(3, 28);
        break;
    case 4:
        call_onekernel(4, 27);
        break;
    case 5:
        call_onekernel(5, 26);
        break;
    case 6:
        call_onekernel(6, 25);
        break;
    case 7:
        call_onekernel(7, 24);
        break;
    case 8:
        call_onekernel(8, 23);
        break;
    case 9:
        call_onekernel(9, 22);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave32(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 32);
        break;
    case 1:
        call_onekernel(1, 31);
        break;
    case 2:
        call_onekernel(2, 30);
        break;
    case 3:
        call_onekernel(3, 29);
        break;
    case 4:
        call_onekernel(4, 28);
        break;
    case 5:
        call_onekernel(5, 27);
        break;
    case 6:
        call_onekernel(6, 26);
        break;
    case 7:
        call_onekernel(7, 25);
        break;
    case 8:
        call_onekernel(8, 24);
        break;
    case 9:
        call_onekernel(9, 23);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave33(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 33);
        break;
    case 1:
        call_onekernel(1, 32);
        break;
    case 2:
        call_onekernel(2, 31);
        break;
    case 3:
        call_onekernel(3, 30);
        break;
    case 4:
        call_onekernel(4, 29);
        break;
    case 5:
        call_onekernel(5, 28);
        break;
    case 6:
        call_onekernel(6, 27);
        break;
    case 7:
        call_onekernel(7, 26);
        break;
    case 8:
        call_onekernel(8, 25);
        break;
    case 9:
        call_onekernel(9, 24);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave34(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 34);
        break;
    case 1:
        call_onekernel(1, 33);
        break;
    case 2:
        call_onekernel(2, 32);
        break;
    case 3:
        call_onekernel(3, 31);
        break;
    case 4:
        call_onekernel(4, 30);
        break;
    case 5:
        call_onekernel(5, 29);
        break;
    case 6:
        call_onekernel(6, 28);
        break;
    case 7:
        call_onekernel(7, 27);
        break;
    case 8:
        call_onekernel(8, 26);
        break;
    case 9:
        call_onekernel(9, 25);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave35(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 35);
        break;
    case 1:
        call_onekernel(1, 34);
        break;
    case 2:
        call_onekernel(2, 33);
        break;
    case 3:
        call_onekernel(3, 32);
        break;
    case 4:
        call_onekernel(4, 31);
        break;
    case 5:
        call_onekernel(5, 30);
        break;
    case 6:
        call_onekernel(6, 29);
        break;
    case 7:
        call_onekernel(7, 28);
        break;
    case 8:
        call_onekernel(8, 27);
        break;
    case 9:
        call_onekernel(9, 26);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave36(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 36);
        break;
    case 1:
        call_onekernel(1, 35);
        break;
    case 2:
        call_onekernel(2, 34);
        break;
    case 3:
        call_onekernel(3, 33);
        break;
    case 4:
        call_onekernel(4, 32);
        break;
    case 5:
        call_onekernel(5, 31);
        break;
    case 6:
        call_onekernel(6, 30);
        break;
    case 7:
        call_onekernel(7, 29);
        break;
    case 8:
        call_onekernel(8, 28);
        break;
    case 9:
        call_onekernel(9, 27);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave37(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 37);
        break;
    case 1:
        call_onekernel(1, 36);
        break;
    case 2:
        call_onekernel(2, 35);
        break;
    case 3:
        call_onekernel(3, 34);
        break;
    case 4:
        call_onekernel(4, 33);
        break;
    case 5:
        call_onekernel(5, 32);
        break;
    case 6:
        call_onekernel(6, 31);
        break;
    case 7:
        call_onekernel(7, 30);
        break;
    case 8:
        call_onekernel(8, 29);
        break;
    case 9:
        call_onekernel(9, 28);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave38(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 38);
        break;
    case 1:
        call_onekernel(1, 37);
        break;
    case 2:
        call_onekernel(2, 36);
        break;
    case 3:
        call_onekernel(3, 35);
        break;
    case 4:
        call_onekernel(4, 34);
        break;
    case 5:
        call_onekernel(5, 33);
        break;
    case 6:
        call_onekernel(6, 32);
        break;
    case 7:
        call_onekernel(7, 31);
        break;
    case 8:
        call_onekernel(8, 30);
        break;
    case 9:
        call_onekernel(9, 29);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave39(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 39);
        break;
    case 1:
        call_onekernel(1, 38);
        break;
    case 2:
        call_onekernel(2, 37);
        break;
    case 3:
        call_onekernel(3, 36);
        break;
    case 4:
        call_onekernel(4, 35);
        break;
    case 5:
        call_onekernel(5, 34);
        break;
    case 6:
        call_onekernel(6, 33);
        break;
    case 7:
        call_onekernel(7, 32);
        break;
    case 8:
        call_onekernel(8, 31);
        break;
    case 9:
        call_onekernel(9, 30);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave40(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 40);
        break;
    case 1:
        call_onekernel(1, 39);
        break;
    case 2:
        call_onekernel(2, 38);
        break;
    case 3:
        call_onekernel(3, 37);
        break;
    case 4:
        call_onekernel(4, 36);
        break;
    case 5:
        call_onekernel(5, 35);
        break;
    case 6:
        call_onekernel(6, 34);
        break;
    case 7:
        call_onekernel(7, 33);
        break;
    case 8:
        call_onekernel(8, 32);
        break;
    case 9:
        call_onekernel(9, 31);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave41(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 41);
        break;
    case 1:
        call_onekernel(1, 40);
        break;
    case 2:
        call_onekernel(2, 39);
        break;
    case 3:
        call_onekernel(3, 38);
        break;
    case 4:
        call_onekernel(4, 37);
        break;
    case 5:
        call_onekernel(5, 36);
        break;
    case 6:
        call_onekernel(6, 35);
        break;
    case 7:
        call_onekernel(7, 34);
        break;
    case 8:
        call_onekernel(8, 33);
        break;
    case 9:
        call_onekernel(9, 32);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave42(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 42);
        break;
    case 1:
        call_onekernel(1, 41);
        break;
    case 2:
        call_onekernel(2, 40);
        break;
    case 3:
        call_onekernel(3, 39);
        break;
    case 4:
        call_onekernel(4, 38);
        break;
    case 5:
        call_onekernel(5, 37);
        break;
    case 6:
        call_onekernel(6, 36);
        break;
    case 7:
        call_onekernel(7, 35);
        break;
    case 8:
        call_onekernel(8, 34);
        break;
    case 9:
        call_onekernel(9, 33);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave43(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 43);
        break;
    case 1:
        call_onekernel(1, 42);
        break;
    case 2:
        call_onekernel(2, 41);
        break;
    case 3:
        call_onekernel(3, 40);
        break;
    case 4:
        call_onekernel(4, 39);
        break;
    case 5:
        call_onekernel(5, 38);
        break;
    case 6:
        call_onekernel(6, 37);
        break;
    case 7:
        call_onekernel(7, 36);
        break;
    case 8:
        call_onekernel(8, 35);
        break;
    case 9:
        call_onekernel(9, 34);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave44(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 44);
        break;
    case 1:
        call_onekernel(1, 43);
        break;
    case 2:
        call_onekernel(2, 42);
        break;
    case 3:
        call_onekernel(3, 41);
        break;
    case 4:
        call_onekernel(4, 40);
        break;
    case 5:
        call_onekernel(5, 39);
        break;
    case 6:
        call_onekernel(6, 38);
        break;
    case 7:
        call_onekernel(7, 37);
        break;
    case 8:
        call_onekernel(8, 36);
        break;
    case 9:
        call_onekernel(9, 35);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave45(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 45);
        break;
    case 1:
        call_onekernel(1, 44);
        break;
    case 2:
        call_onekernel(2, 43);
        break;
    case 3:
        call_onekernel(3, 42);
        break;
    case 4:
        call_onekernel(4, 41);
        break;
    case 5:
        call_onekernel(5, 40);
        break;
    case 6:
        call_onekernel(6, 39);
        break;
    case 7:
        call_onekernel(7, 38);
        break;
    case 8:
        call_onekernel(8, 37);
        break;
    case 9:
        call_onekernel(9, 36);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave46(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 46);
        break;
    case 1:
        call_onekernel(1, 45);
        break;
    case 2:
        call_onekernel(2, 44);
        break;
    case 3:
        call_onekernel(3, 43);
        break;
    case 4:
        call_onekernel(4, 42);
        break;
    case 5:
        call_onekernel(5, 41);
        break;
    case 6:
        call_onekernel(6, 40);
        break;
    case 7:
        call_onekernel(7, 39);
        break;
    case 8:
        call_onekernel(8, 38);
        break;
    case 9:
        call_onekernel(9, 37);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave47(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 47);
        break;
    case 1:
        call_onekernel(1, 46);
        break;
    case 2:
        call_onekernel(2, 45);
        break;
    case 3:
        call_onekernel(3, 44);
        break;
    case 4:
        call_onekernel(4, 43);
        break;
    case 5:
        call_onekernel(5, 42);
        break;
    case 6:
        call_onekernel(6, 41);
        break;
    case 7:
        call_onekernel(7, 40);
        break;
    case 8:
        call_onekernel(8, 39);
        break;
    case 9:
        call_onekernel(9, 38);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave48(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 48);
        break;
    case 1:
        call_onekernel(1, 47);
        break;
    case 2:
        call_onekernel(2, 46);
        break;
    case 3:
        call_onekernel(3, 45);
        break;
    case 4:
        call_onekernel(4, 44);
        break;
    case 5:
        call_onekernel(5, 43);
        break;
    case 6:
        call_onekernel(6, 42);
        break;
    case 7:
        call_onekernel(7, 41);
        break;
    case 8:
        call_onekernel(8, 40);
        break;
    case 9:
        call_onekernel(9, 39);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave49(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 49);
        break;
    case 1:
        call_onekernel(1, 48);
        break;
    case 2:
        call_onekernel(2, 47);
        break;
    case 3:
        call_onekernel(3, 46);
        break;
    case 4:
        call_onekernel(4, 45);
        break;
    case 5:
        call_onekernel(5, 44);
        break;
    case 6:
        call_onekernel(6, 43);
        break;
    case 7:
        call_onekernel(7, 42);
        break;
    case 8:
        call_onekernel(8, 41);
        break;
    case 9:
        call_onekernel(9, 40);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave50(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 50);
        break;
    case 1:
        call_onekernel(1, 49);
        break;
    case 2:
        call_onekernel(2, 48);
        break;
    case 3:
        call_onekernel(3, 47);
        break;
    case 4:
        call_onekernel(4, 46);
        break;
    case 5:
        call_onekernel(5, 45);
        break;
    case 6:
        call_onekernel(6, 44);
        break;
    case 7:
        call_onekernel(7, 43);
        break;
    case 8:
        call_onekernel(8, 42);
        break;
    case 9:
        call_onekernel(9, 41);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave51(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 51);
        break;
    case 1:
        call_onekernel(1, 50);
        break;
    case 2:
        call_onekernel(2, 49);
        break;
    case 3:
        call_onekernel(3, 48);
        break;
    case 4:
        call_onekernel(4, 47);
        break;
    case 5:
        call_onekernel(5, 46);
        break;
    case 6:
        call_onekernel(6, 45);
        break;
    case 7:
        call_onekernel(7, 44);
        break;
    case 8:
        call_onekernel(8, 43);
        break;
    case 9:
        call_onekernel(9, 42);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave52(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 52);
        break;
    case 1:
        call_onekernel(1, 51);
        break;
    case 2:
        call_onekernel(2, 50);
        break;
    case 3:
        call_onekernel(3, 49);
        break;
    case 4:
        call_onekernel(4, 48);
        break;
    case 5:
        call_onekernel(5, 47);
        break;
    case 6:
        call_onekernel(6, 46);
        break;
    case 7:
        call_onekernel(7, 45);
        break;
    case 8:
        call_onekernel(8, 44);
        break;
    case 9:
        call_onekernel(9, 43);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave53(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 53);
        break;
    case 1:
        call_onekernel(1, 52);
        break;
    case 2:
        call_onekernel(2, 51);
        break;
    case 3:
        call_onekernel(3, 50);
        break;
    case 4:
        call_onekernel(4, 49);
        break;
    case 5:
        call_onekernel(5, 48);
        break;
    case 6:
        call_onekernel(6, 47);
        break;
    case 7:
        call_onekernel(7, 46);
        break;
    case 8:
        call_onekernel(8, 45);
        break;
    case 9:
        call_onekernel(9, 44);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave54(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 54);
        break;
    case 1:
        call_onekernel(1, 53);
        break;
    case 2:
        call_onekernel(2, 52);
        break;
    case 3:
        call_onekernel(3, 51);
        break;
    case 4:
        call_onekernel(4, 50);
        break;
    case 5:
        call_onekernel(5, 49);
        break;
    case 6:
        call_onekernel(6, 48);
        break;
    case 7:
        call_onekernel(7, 47);
        break;
    case 8:
        call_onekernel(8, 46);
        break;
    case 9:
        call_onekernel(9, 45);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave55(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 55);
        break;
    case 1:
        call_onekernel(1, 54);
        break;
    case 2:
        call_onekernel(2, 53);
        break;
    case 3:
        call_onekernel(3, 52);
        break;
    case 4:
        call_onekernel(4, 51);
        break;
    case 5:
        call_onekernel(5, 50);
        break;
    case 6:
        call_onekernel(6, 49);
        break;
    case 7:
        call_onekernel(7, 48);
        break;
    case 8:
        call_onekernel(8, 47);
        break;
    case 9:
        call_onekernel(9, 46);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave56(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 56);
        break;
    case 1:
        call_onekernel(1, 55);
        break;
    case 2:
        call_onekernel(2, 54);
        break;
    case 3:
        call_onekernel(3, 53);
        break;
    case 4:
        call_onekernel(4, 52);
        break;
    case 5:
        call_onekernel(5, 51);
        break;
    case 6:
        call_onekernel(6, 50);
        break;
    case 7:
        call_onekernel(7, 49);
        break;
    case 8:
        call_onekernel(8, 48);
        break;
    case 9:
        call_onekernel(9, 47);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave57(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 57);
        break;
    case 1:
        call_onekernel(1, 56);
        break;
    case 2:
        call_onekernel(2, 55);
        break;
    case 3:
        call_onekernel(3, 54);
        break;
    case 4:
        call_onekernel(4, 53);
        break;
    case 5:
        call_onekernel(5, 52);
        break;
    case 6:
        call_onekernel(6, 51);
        break;
    case 7:
        call_onekernel(7, 50);
        break;
    case 8:
        call_onekernel(8, 49);
        break;
    case 9:
        call_onekernel(9, 48);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave58(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 58);
        break;
    case 1:
        call_onekernel(1, 57);
        break;
    case 2:
        call_onekernel(2, 56);
        break;
    case 3:
        call_onekernel(3, 55);
        break;
    case 4:
        call_onekernel(4, 54);
        break;
    case 5:
        call_onekernel(5, 53);
        break;
    case 6:
        call_onekernel(6, 52);
        break;
    case 7:
        call_onekernel(7, 51);
        break;
    case 8:
        call_onekernel(8, 50);
        break;
    case 9:
        call_onekernel(9, 49);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave59(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 59);
        break;
    case 1:
        call_onekernel(1, 58);
        break;
    case 2:
        call_onekernel(2, 57);
        break;
    case 3:
        call_onekernel(3, 56);
        break;
    case 4:
        call_onekernel(4, 55);
        break;
    case 5:
        call_onekernel(5, 54);
        break;
    case 6:
        call_onekernel(6, 53);
        break;
    case 7:
        call_onekernel(7, 52);
        break;
    case 8:
        call_onekernel(8, 51);
        break;
    case 9:
        call_onekernel(9, 50);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave60(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 60);
        break;
    case 1:
        call_onekernel(1, 59);
        break;
    case 2:
        call_onekernel(2, 58);
        break;
    case 3:
        call_onekernel(3, 57);
        break;
    case 4:
        call_onekernel(4, 56);
        break;
    case 5:
        call_onekernel(5, 55);
        break;
    case 6:
        call_onekernel(6, 54);
        break;
    case 7:
        call_onekernel(7, 53);
        break;
    case 8:
        call_onekernel(8, 52);
        break;
    case 9:
        call_onekernel(9, 51);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave61(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 61);
        break;
    case 1:
        call_onekernel(1, 60);
        break;
    case 2:
        call_onekernel(2, 59);
        break;
    case 3:
        call_onekernel(3, 58);
        break;
    case 4:
        call_onekernel(4, 57);
        break;
    case 5:
        call_onekernel(5, 56);
        break;
    case 6:
        call_onekernel(6, 55);
        break;
    case 7:
        call_onekernel(7, 54);
        break;
    case 8:
        call_onekernel(8, 53);
        break;
    case 9:
        call_onekernel(9, 52);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave62(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 62);
        break;
    case 1:
        call_onekernel(1, 61);
        break;
    case 2:
        call_onekernel(2, 60);
        break;
    case 3:
        call_onekernel(3, 59);
        break;
    case 4:
        call_onekernel(4, 58);
        break;
    case 5:
        call_onekernel(5, 57);
        break;
    case 6:
        call_onekernel(6, 56);
        break;
    case 7:
        call_onekernel(7, 55);
        break;
    case 8:
        call_onekernel(8, 54);
        break;
    case 9:
        call_onekernel(9, 53);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave63(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 63);
        break;
    case 1:
        call_onekernel(1, 62);
        break;
    case 2:
        call_onekernel(2, 61);
        break;
    case 3:
        call_onekernel(3, 60);
        break;
    case 4:
        call_onekernel(4, 59);
        break;
    case 5:
        call_onekernel(5, 58);
        break;
    case 6:
        call_onekernel(6, 57);
        break;
    case 7:
        call_onekernel(7, 56);
        break;
    case 8:
        call_onekernel(8, 55);
        break;
    case 9:
        call_onekernel(9, 54);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave64(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 64);
        break;
    case 1:
        call_onekernel(1, 63);
        break;
    case 2:
        call_onekernel(2, 62);
        break;
    case 3:
        call_onekernel(3, 61);
        break;
    case 4:
        call_onekernel(4, 60);
        break;
    case 5:
        call_onekernel(5, 59);
        break;
    case 6:
        call_onekernel(6, 58);
        break;
    case 7:
        call_onekernel(7, 57);
        break;
    case 8:
        call_onekernel(8, 56);
        break;
    case 9:
        call_onekernel(9, 55);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave65(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 65);
        break;
    case 1:
        call_onekernel(1, 64);
        break;
    case 2:
        call_onekernel(2, 63);
        break;
    case 3:
        call_onekernel(3, 62);
        break;
    case 4:
        call_onekernel(4, 61);
        break;
    case 5:
        call_onekernel(5, 60);
        break;
    case 6:
        call_onekernel(6, 59);
        break;
    case 7:
        call_onekernel(7, 58);
        break;
    case 8:
        call_onekernel(8, 57);
        break;
    case 9:
        call_onekernel(9, 56);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave66(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 66);
        break;
    case 1:
        call_onekernel(1, 65);
        break;
    case 2:
        call_onekernel(2, 64);
        break;
    case 3:
        call_onekernel(3, 63);
        break;
    case 4:
        call_onekernel(4, 62);
        break;
    case 5:
        call_onekernel(5, 61);
        break;
    case 6:
        call_onekernel(6, 60);
        break;
    case 7:
        call_onekernel(7, 59);
        break;
    case 8:
        call_onekernel(8, 58);
        break;
    case 9:
        call_onekernel(9, 57);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave67(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 67);
        break;
    case 1:
        call_onekernel(1, 66);
        break;
    case 2:
        call_onekernel(2, 65);
        break;
    case 3:
        call_onekernel(3, 64);
        break;
    case 4:
        call_onekernel(4, 63);
        break;
    case 5:
        call_onekernel(5, 62);
        break;
    case 6:
        call_onekernel(6, 61);
        break;
    case 7:
        call_onekernel(7, 60);
        break;
    case 8:
        call_onekernel(8, 59);
        break;
    case 9:
        call_onekernel(9, 58);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave68(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 68);
        break;
    case 1:
        call_onekernel(1, 67);
        break;
    case 2:
        call_onekernel(2, 66);
        break;
    case 3:
        call_onekernel(3, 65);
        break;
    case 4:
        call_onekernel(4, 64);
        break;
    case 5:
        call_onekernel(5, 63);
        break;
    case 6:
        call_onekernel(6, 62);
        break;
    case 7:
        call_onekernel(7, 61);
        break;
    case 8:
        call_onekernel(8, 60);
        break;
    case 9:
        call_onekernel(9, 59);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave69(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 69);
        break;
    case 1:
        call_onekernel(1, 68);
        break;
    case 2:
        call_onekernel(2, 67);
        break;
    case 3:
        call_onekernel(3, 66);
        break;
    case 4:
        call_onekernel(4, 65);
        break;
    case 5:
        call_onekernel(5, 64);
        break;
    case 6:
        call_onekernel(6, 63);
        break;
    case 7:
        call_onekernel(7, 62);
        break;
    case 8:
        call_onekernel(8, 61);
        break;
    case 9:
        call_onekernel(9, 60);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave70(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 70);
        break;
    case 1:
        call_onekernel(1, 69);
        break;
    case 2:
        call_onekernel(2, 68);
        break;
    case 3:
        call_onekernel(3, 67);
        break;
    case 4:
        call_onekernel(4, 66);
        break;
    case 5:
        call_onekernel(5, 65);
        break;
    case 6:
        call_onekernel(6, 64);
        break;
    case 7:
        call_onekernel(7, 63);
        break;
    case 8:
        call_onekernel(8, 62);
        break;
    case 9:
        call_onekernel(9, 61);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave71(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 71);
        break;
    case 1:
        call_onekernel(1, 70);
        break;
    case 2:
        call_onekernel(2, 69);
        break;
    case 3:
        call_onekernel(3, 68);
        break;
    case 4:
        call_onekernel(4, 67);
        break;
    case 5:
        call_onekernel(5, 66);
        break;
    case 6:
        call_onekernel(6, 65);
        break;
    case 7:
        call_onekernel(7, 64);
        break;
    case 8:
        call_onekernel(8, 63);
        break;
    case 9:
        call_onekernel(9, 62);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave72(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 72);
        break;
    case 1:
        call_onekernel(1, 71);
        break;
    case 2:
        call_onekernel(2, 70);
        break;
    case 3:
        call_onekernel(3, 69);
        break;
    case 4:
        call_onekernel(4, 68);
        break;
    case 5:
        call_onekernel(5, 67);
        break;
    case 6:
        call_onekernel(6, 66);
        break;
    case 7:
        call_onekernel(7, 65);
        break;
    case 8:
        call_onekernel(8, 64);
        break;
    case 9:
        call_onekernel(9, 63);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave73(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 73);
        break;
    case 1:
        call_onekernel(1, 72);
        break;
    case 2:
        call_onekernel(2, 71);
        break;
    case 3:
        call_onekernel(3, 70);
        break;
    case 4:
        call_onekernel(4, 69);
        break;
    case 5:
        call_onekernel(5, 68);
        break;
    case 6:
        call_onekernel(6, 67);
        break;
    case 7:
        call_onekernel(7, 66);
        break;
    case 8:
        call_onekernel(8, 65);
        break;
    case 9:
        call_onekernel(9, 64);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave74(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 74);
        break;
    case 1:
        call_onekernel(1, 73);
        break;
    case 2:
        call_onekernel(2, 72);
        break;
    case 3:
        call_onekernel(3, 71);
        break;
    case 4:
        call_onekernel(4, 70);
        break;
    case 5:
        call_onekernel(5, 69);
        break;
    case 6:
        call_onekernel(6, 68);
        break;
    case 7:
        call_onekernel(7, 67);
        break;
    case 8:
        call_onekernel(8, 66);
        break;
    case 9:
        call_onekernel(9, 65);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave75(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 75);
        break;
    case 1:
        call_onekernel(1, 74);
        break;
    case 2:
        call_onekernel(2, 73);
        break;
    case 3:
        call_onekernel(3, 72);
        break;
    case 4:
        call_onekernel(4, 71);
        break;
    case 5:
        call_onekernel(5, 70);
        break;
    case 6:
        call_onekernel(6, 69);
        break;
    case 7:
        call_onekernel(7, 68);
        break;
    case 8:
        call_onekernel(8, 67);
        break;
    case 9:
        call_onekernel(9, 66);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave76(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 76);
        break;
    case 1:
        call_onekernel(1, 75);
        break;
    case 2:
        call_onekernel(2, 74);
        break;
    case 3:
        call_onekernel(3, 73);
        break;
    case 4:
        call_onekernel(4, 72);
        break;
    case 5:
        call_onekernel(5, 71);
        break;
    case 6:
        call_onekernel(6, 70);
        break;
    case 7:
        call_onekernel(7, 69);
        break;
    case 8:
        call_onekernel(8, 68);
        break;
    case 9:
        call_onekernel(9, 67);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave77(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 77);
        break;
    case 1:
        call_onekernel(1, 76);
        break;
    case 2:
        call_onekernel(2, 75);
        break;
    case 3:
        call_onekernel(3, 74);
        break;
    case 4:
        call_onekernel(4, 73);
        break;
    case 5:
        call_onekernel(5, 72);
        break;
    case 6:
        call_onekernel(6, 71);
        break;
    case 7:
        call_onekernel(7, 70);
        break;
    case 8:
        call_onekernel(8, 69);
        break;
    case 9:
        call_onekernel(9, 68);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave78(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 78);
        break;
    case 1:
        call_onekernel(1, 77);
        break;
    case 2:
        call_onekernel(2, 76);
        break;
    case 3:
        call_onekernel(3, 75);
        break;
    case 4:
        call_onekernel(4, 74);
        break;
    case 5:
        call_onekernel(5, 73);
        break;
    case 6:
        call_onekernel(6, 72);
        break;
    case 7:
        call_onekernel(7, 71);
        break;
    case 8:
        call_onekernel(8, 70);
        break;
    case 9:
        call_onekernel(9, 69);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave79(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 79);
        break;
    case 1:
        call_onekernel(1, 78);
        break;
    case 2:
        call_onekernel(2, 77);
        break;
    case 3:
        call_onekernel(3, 76);
        break;
    case 4:
        call_onekernel(4, 75);
        break;
    case 5:
        call_onekernel(5, 74);
        break;
    case 6:
        call_onekernel(6, 73);
        break;
    case 7:
        call_onekernel(7, 72);
        break;
    case 8:
        call_onekernel(8, 71);
        break;
    case 9:
        call_onekernel(9, 70);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave80(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 80);
        break;
    case 1:
        call_onekernel(1, 79);
        break;
    case 2:
        call_onekernel(2, 78);
        break;
    case 3:
        call_onekernel(3, 77);
        break;
    case 4:
        call_onekernel(4, 76);
        break;
    case 5:
        call_onekernel(5, 75);
        break;
    case 6:
        call_onekernel(6, 74);
        break;
    case 7:
        call_onekernel(7, 73);
        break;
    case 8:
        call_onekernel(8, 72);
        break;
    case 9:
        call_onekernel(9, 71);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave81(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 81);
        break;
    case 1:
        call_onekernel(1, 80);
        break;
    case 2:
        call_onekernel(2, 79);
        break;
    case 3:
        call_onekernel(3, 78);
        break;
    case 4:
        call_onekernel(4, 77);
        break;
    case 5:
        call_onekernel(5, 76);
        break;
    case 6:
        call_onekernel(6, 75);
        break;
    case 7:
        call_onekernel(7, 74);
        break;
    case 8:
        call_onekernel(8, 73);
        break;
    case 9:
        call_onekernel(9, 72);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave82(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 82);
        break;
    case 1:
        call_onekernel(1, 81);
        break;
    case 2:
        call_onekernel(2, 80);
        break;
    case 3:
        call_onekernel(3, 79);
        break;
    case 4:
        call_onekernel(4, 78);
        break;
    case 5:
        call_onekernel(5, 77);
        break;
    case 6:
        call_onekernel(6, 76);
        break;
    case 7:
        call_onekernel(7, 75);
        break;
    case 8:
        call_onekernel(8, 74);
        break;
    case 9:
        call_onekernel(9, 73);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave83(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 83);
        break;
    case 1:
        call_onekernel(1, 82);
        break;
    case 2:
        call_onekernel(2, 81);
        break;
    case 3:
        call_onekernel(3, 80);
        break;
    case 4:
        call_onekernel(4, 79);
        break;
    case 5:
        call_onekernel(5, 78);
        break;
    case 6:
        call_onekernel(6, 77);
        break;
    case 7:
        call_onekernel(7, 76);
        break;
    case 8:
        call_onekernel(8, 75);
        break;
    case 9:
        call_onekernel(9, 74);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave84(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 84);
        break;
    case 1:
        call_onekernel(1, 83);
        break;
    case 2:
        call_onekernel(2, 82);
        break;
    case 3:
        call_onekernel(3, 81);
        break;
    case 4:
        call_onekernel(4, 80);
        break;
    case 5:
        call_onekernel(5, 79);
        break;
    case 6:
        call_onekernel(6, 78);
        break;
    case 7:
        call_onekernel(7, 77);
        break;
    case 8:
        call_onekernel(8, 76);
        break;
    case 9:
        call_onekernel(9, 75);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave85(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 85);
        break;
    case 1:
        call_onekernel(1, 84);
        break;
    case 2:
        call_onekernel(2, 83);
        break;
    case 3:
        call_onekernel(3, 82);
        break;
    case 4:
        call_onekernel(4, 81);
        break;
    case 5:
        call_onekernel(5, 80);
        break;
    case 6:
        call_onekernel(6, 79);
        break;
    case 7:
        call_onekernel(7, 78);
        break;
    case 8:
        call_onekernel(8, 77);
        break;
    case 9:
        call_onekernel(9, 76);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave86(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 86);
        break;
    case 1:
        call_onekernel(1, 85);
        break;
    case 2:
        call_onekernel(2, 84);
        break;
    case 3:
        call_onekernel(3, 83);
        break;
    case 4:
        call_onekernel(4, 82);
        break;
    case 5:
        call_onekernel(5, 81);
        break;
    case 6:
        call_onekernel(6, 80);
        break;
    case 7:
        call_onekernel(7, 79);
        break;
    case 8:
        call_onekernel(8, 78);
        break;
    case 9:
        call_onekernel(9, 77);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave87(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 87);
        break;
    case 1:
        call_onekernel(1, 86);
        break;
    case 2:
        call_onekernel(2, 85);
        break;
    case 3:
        call_onekernel(3, 84);
        break;
    case 4:
        call_onekernel(4, 83);
        break;
    case 5:
        call_onekernel(5, 82);
        break;
    case 6:
        call_onekernel(6, 81);
        break;
    case 7:
        call_onekernel(7, 80);
        break;
    case 8:
        call_onekernel(8, 79);
        break;
    case 9:
        call_onekernel(9, 78);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave88(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 88);
        break;
    case 1:
        call_onekernel(1, 87);
        break;
    case 2:
        call_onekernel(2, 86);
        break;
    case 3:
        call_onekernel(3, 85);
        break;
    case 4:
        call_onekernel(4, 84);
        break;
    case 5:
        call_onekernel(5, 83);
        break;
    case 6:
        call_onekernel(6, 82);
        break;
    case 7:
        call_onekernel(7, 81);
        break;
    case 8:
        call_onekernel(8, 80);
        break;
    case 9:
        call_onekernel(9, 79);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave89(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 89);
        break;
    case 1:
        call_onekernel(1, 88);
        break;
    case 2:
        call_onekernel(2, 87);
        break;
    case 3:
        call_onekernel(3, 86);
        break;
    case 4:
        call_onekernel(4, 85);
        break;
    case 5:
        call_onekernel(5, 84);
        break;
    case 6:
        call_onekernel(6, 83);
        break;
    case 7:
        call_onekernel(7, 82);
        break;
    case 8:
        call_onekernel(8, 81);
        break;
    case 9:
        call_onekernel(9, 80);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave90(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 90);
        break;
    case 1:
        call_onekernel(1, 89);
        break;
    case 2:
        call_onekernel(2, 88);
        break;
    case 3:
        call_onekernel(3, 87);
        break;
    case 4:
        call_onekernel(4, 86);
        break;
    case 5:
        call_onekernel(5, 85);
        break;
    case 6:
        call_onekernel(6, 84);
        break;
    case 7:
        call_onekernel(7, 83);
        break;
    case 8:
        call_onekernel(8, 82);
        break;
    case 9:
        call_onekernel(9, 81);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave91(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 91);
        break;
    case 1:
        call_onekernel(1, 90);
        break;
    case 2:
        call_onekernel(2, 89);
        break;
    case 3:
        call_onekernel(3, 88);
        break;
    case 4:
        call_onekernel(4, 87);
        break;
    case 5:
        call_onekernel(5, 86);
        break;
    case 6:
        call_onekernel(6, 85);
        break;
    case 7:
        call_onekernel(7, 84);
        break;
    case 8:
        call_onekernel(8, 83);
        break;
    case 9:
        call_onekernel(9, 82);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave92(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 92);
        break;
    case 1:
        call_onekernel(1, 91);
        break;
    case 2:
        call_onekernel(2, 90);
        break;
    case 3:
        call_onekernel(3, 89);
        break;
    case 4:
        call_onekernel(4, 88);
        break;
    case 5:
        call_onekernel(5, 87);
        break;
    case 6:
        call_onekernel(6, 86);
        break;
    case 7:
        call_onekernel(7, 85);
        break;
    case 8:
        call_onekernel(8, 84);
        break;
    case 9:
        call_onekernel(9, 83);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave93(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 93);
        break;
    case 1:
        call_onekernel(1, 92);
        break;
    case 2:
        call_onekernel(2, 91);
        break;
    case 3:
        call_onekernel(3, 90);
        break;
    case 4:
        call_onekernel(4, 89);
        break;
    case 5:
        call_onekernel(5, 88);
        break;
    case 6:
        call_onekernel(6, 87);
        break;
    case 7:
        call_onekernel(7, 86);
        break;
    case 8:
        call_onekernel(8, 85);
        break;
    case 9:
        call_onekernel(9, 84);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave94(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 94);
        break;
    case 1:
        call_onekernel(1, 93);
        break;
    case 2:
        call_onekernel(2, 92);
        break;
    case 3:
        call_onekernel(3, 91);
        break;
    case 4:
        call_onekernel(4, 90);
        break;
    case 5:
        call_onekernel(5, 89);
        break;
    case 6:
        call_onekernel(6, 88);
        break;
    case 7:
        call_onekernel(7, 87);
        break;
    case 8:
        call_onekernel(8, 86);
        break;
    case 9:
        call_onekernel(9, 85);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave95(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 95);
        break;
    case 1:
        call_onekernel(1, 94);
        break;
    case 2:
        call_onekernel(2, 93);
        break;
    case 3:
        call_onekernel(3, 92);
        break;
    case 4:
        call_onekernel(4, 91);
        break;
    case 5:
        call_onekernel(5, 90);
        break;
    case 6:
        call_onekernel(6, 89);
        break;
    case 7:
        call_onekernel(7, 88);
        break;
    case 8:
        call_onekernel(8, 87);
        break;
    case 9:
        call_onekernel(9, 86);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave96(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 96);
        break;
    case 1:
        call_onekernel(1, 95);
        break;
    case 2:
        call_onekernel(2, 94);
        break;
    case 3:
        call_onekernel(3, 93);
        break;
    case 4:
        call_onekernel(4, 92);
        break;
    case 5:
        call_onekernel(5, 91);
        break;
    case 6:
        call_onekernel(6, 90);
        break;
    case 7:
        call_onekernel(7, 89);
        break;
    case 8:
        call_onekernel(8, 88);
        break;
    case 9:
        call_onekernel(9, 87);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave97(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 97);
        break;
    case 1:
        call_onekernel(1, 96);
        break;
    case 2:
        call_onekernel(2, 95);
        break;
    case 3:
        call_onekernel(3, 94);
        break;
    case 4:
        call_onekernel(4, 93);
        break;
    case 5:
        call_onekernel(5, 92);
        break;
    case 6:
        call_onekernel(6, 91);
        break;
    case 7:
        call_onekernel(7, 90);
        break;
    case 8:
        call_onekernel(8, 89);
        break;
    case 9:
        call_onekernel(9, 88);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave98(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 98);
        break;
    case 1:
        call_onekernel(1, 97);
        break;
    case 2:
        call_onekernel(2, 96);
        break;
    case 3:
        call_onekernel(3, 95);
        break;
    case 4:
        call_onekernel(4, 94);
        break;
    case 5:
        call_onekernel(5, 93);
        break;
    case 6:
        call_onekernel(6, 92);
        break;
    case 7:
        call_onekernel(7, 91);
        break;
    case 8:
        call_onekernel(8, 90);
        break;
    case 9:
        call_onekernel(9, 89);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave99(WaveInputParamsBS4 *__restrict__ input,
           WaveModelParamsBS4 *__restrict__ model,
           WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(0, 99);
        break;
    case 1:
        call_onekernel(1, 98);
        break;
    case 2:
        call_onekernel(2, 97);
        break;
    case 3:
        call_onekernel(3, 96);
        break;
    case 4:
        call_onekernel(4, 95);
        break;
    case 5:
        call_onekernel(5, 94);
        break;
    case 6:
        call_onekernel(6, 93);
        break;
    case 7:
        call_onekernel(7, 92);
        break;
    case 8:
        call_onekernel(8, 91);
        break;
    case 9:
        call_onekernel(9, 90);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave100(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(1, 99);
        break;
    case 1:
        call_onekernel(2, 98);
        break;
    case 2:
        call_onekernel(3, 97);
        break;
    case 3:
        call_onekernel(4, 96);
        break;
    case 4:
        call_onekernel(5, 95);
        break;
    case 5:
        call_onekernel(6, 94);
        break;
    case 6:
        call_onekernel(7, 93);
        break;
    case 7:
        call_onekernel(8, 92);
        break;
    case 8:
        call_onekernel(9, 91);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave101(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(2, 99);
        break;
    case 1:
        call_onekernel(3, 98);
        break;
    case 2:
        call_onekernel(4, 97);
        break;
    case 3:
        call_onekernel(5, 96);
        break;
    case 4:
        call_onekernel(6, 95);
        break;
    case 5:
        call_onekernel(7, 94);
        break;
    case 6:
        call_onekernel(8, 93);
        break;
    case 7:
        call_onekernel(9, 92);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave102(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(3, 99);
        break;
    case 1:
        call_onekernel(4, 98);
        break;
    case 2:
        call_onekernel(5, 97);
        break;
    case 3:
        call_onekernel(6, 96);
        break;
    case 4:
        call_onekernel(7, 95);
        break;
    case 5:
        call_onekernel(8, 94);
        break;
    case 6:
        call_onekernel(9, 93);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave103(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(4, 99);
        break;
    case 1:
        call_onekernel(5, 98);
        break;
    case 2:
        call_onekernel(6, 97);
        break;
    case 3:
        call_onekernel(7, 96);
        break;
    case 4:
        call_onekernel(8, 95);
        break;
    case 5:
        call_onekernel(9, 94);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave104(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(5, 99);
        break;
    case 1:
        call_onekernel(6, 98);
        break;
    case 2:
        call_onekernel(7, 97);
        break;
    case 3:
        call_onekernel(8, 96);
        break;
    case 4:
        call_onekernel(9, 95);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave105(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(6, 99);
        break;
    case 1:
        call_onekernel(7, 98);
        break;
    case 2:
        call_onekernel(8, 97);
        break;
    case 3:
        call_onekernel(9, 96);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave106(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(7, 99);
        break;
    case 1:
        call_onekernel(8, 98);
        break;
    case 2:
        call_onekernel(9, 97);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave107(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(8, 99);
        break;
    case 1:
        call_onekernel(9, 98);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave108(WaveInputParamsBS4 *__restrict__ input,
            WaveModelParamsBS4 *__restrict__ model,
            WaveOutputParamsBS4 *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel(9, 99);
        break;
    }
}
