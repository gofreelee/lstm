

enum class GemvParams {
    kColumsPerBlock_32 = 32,
    kThreadNumPerBlock_256 = 256,
    kHiddenSize_256 = 256,
    kInputSize_256 = 256
};

#include <stdio.h>
__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__global__ void gemv(const float *__restrict__ input,
                     const float *__restrict__ weight,
                     float *__restrict__ output) {
    __shared__ float nndense_output[kColumsPerBlock];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx.x * kColumsPerBlock + lane_id;
    nndense_output[lane_id] = 0.0000f;
    float temp = 0.0000f;
    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
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
template <int kColumsPerBlock, int kHiddenSize, int kInputSize,
          int kThreadNumPerBlock>
__global__ void gem4v(const float *__restrict__ input,
                      const float4 *__restrict__ weight,
                      float4 *__restrict__ output) {
    __shared__ float4 nndense_output[kColumsPerBlock];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int colOffset = blockIdx.x * kColumsPerBlock + lane_id;
    nndense_output[lane_id] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    float temp[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    const int ROWS = kInputSize / (kThreadNumPerBlock / 32);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + blockIdx.x * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow) {
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

template __global__ void
gemv<static_cast<int>(GemvParams::kColumsPerBlock_32),
     static_cast<int>(GemvParams::kHiddenSize_256),
     static_cast<int>(GemvParams::kInputSize_256),
     static_cast<int>(GemvParams::kThreadNumPerBlock_256)>(
    const float *__restrict__ input, const float *__restrict__ weight,
    float *__restrict__ output);

template __global__ void
gem4v<static_cast<int>(GemvParams::kColumsPerBlock_32),
      static_cast<int>(GemvParams::kHiddenSize_256),
      static_cast<int>(GemvParams::kInputSize_256),
      static_cast<int>(GemvParams::kThreadNumPerBlock_256)>(
    const float *__restrict__ input, const float4 *__restrict__ weight,
    float4 *__restrict__ output);
