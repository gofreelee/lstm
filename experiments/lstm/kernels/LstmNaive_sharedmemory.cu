__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}
#include <stdio.h>
template <unsigned int hidden_size>
__global__ void
compute_x(const float *__restrict__ data, const float *__restrict__ state_c,
          const float *__restrict__ state_h, const float *__restrict__ W0,
          const float *__restrict__ U0, const float *__restrict__ bias0,
          float *__restrict__ sum_cached_ptr) {
    float x;
    extern __shared__ float sum_cached[];

    // printf("%f\n", sum_cached_ptr[0]);
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x];
    float private_vector_element1 = state_h[threadIdx.x];

    x = W0[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    x += U0[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        x += __shfl_down_sync(0xffffffff, x, i, 32);
    if (lane_id == 0) {
        sum_cached[warp_id] = x;
    }
    // y
    // z
    // k
    __syncthreads();

    x = (warp_id == 0 && lane_id < num_warp) ? sum_cached[lane_id] : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 0 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            x += __shfl_down_sync(0x0ff, x, i, 32); // FIX
        if (lane_id == 0) {
            sum_cached_ptr[blockIdx.x * 32 + 0] = x + bias0[relative_idx];
        }
    }
    if (blockIdx.x < 256) {
        __syncthreads();
    } else {
        float tmp = 0.0;
    }
    // global barrier
    if (blockIdx.x < 512) {
        __syncthreads();
    }
    // global barrier
    float y;
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x];
    float private_vector_element1 = state_h[threadIdx.x];

    y = W1[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    y += U1[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        y += __shfl_down_sync(0xffffffff, y, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[blockIdx.x * 32 + num_warp + warp_id] = y;
    __syncthreads();

    y = warp_id == 1 && lane_id < num_warp
            ? sum_cached_ptr[blockIdx.x * 32 + num_warp + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 1 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            y += __shfl_down_sync(0x0ff, y, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[blockIdx.x * 32 + 1] = y + bias1[relative_idx];
    }
}
//  [0,    ....  31]
//  []  [] [] [] []

template <unsigned int hidden_size>
__global__ void
compute_y(const float *__restrict__ data, const float *__restrict__ state_c,
          const float *__restrict__ state_h, const float *__restrict__ W1,
          const float *__restrict__ U1, const float *__restrict__ bias1,
          float *__restrict__ sum_cached_ptr) {
    float y;
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x];
    float private_vector_element1 = state_h[threadIdx.x];

    y = W1[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    y += U1[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        y += __shfl_down_sync(0xffffffff, y, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[blockIdx.x * 32 + num_warp + warp_id] = y;
    __syncthreads();

    y = warp_id == 1 && lane_id < num_warp
            ? sum_cached_ptr[blockIdx.x * 32 + num_warp + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 1 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            y += __shfl_down_sync(0x0ff, y, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[blockIdx.x * 32 + 1] = y + bias1[relative_idx];
    }
}

template <unsigned int hidden_size>
__global__ void
compute_z(const float *__restrict__ data, const float *__restrict__ state_c,
          const float *__restrict__ state_h, const float *__restrict__ W2,
          const float *__restrict__ U2, const float *__restrict__ bias2,
          float *__restrict__ sum_cached_ptr) {
    float z;
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x];
    float private_vector_element1 = state_h[threadIdx.x];

    z = W2[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    z += U2[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        z += __shfl_down_sync(0xffffffff, z, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[blockIdx.x * 32 + (num_warp << 1) + warp_id] = z;
    __syncthreads();

    z = warp_id == 2 && lane_id < num_warp
            ? sum_cached_ptr[blockIdx.x * 32 + (num_warp << 1) + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 2 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            z += __shfl_down_sync(0x0ff, z, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[blockIdx.x * 32 + 2] = z + bias2[relative_idx];
    }
}

template <unsigned int hidden_size>
__global__ void
compute_k(const float *__restrict__ data, const float *__restrict__ state_c,
          const float *__restrict__ state_h, const float *__restrict__ W3,
          const float *__restrict__ U3, const float *__restrict__ bias3,
          float *__restrict__ sum_cached_ptr) {
    // extern __shared__ float sum_cached[];
    float k;
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x];
    float private_vector_element1 = state_h[threadIdx.x];

    k = W3[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    k += U3[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        k += __shfl_down_sync(0xffffffff, k, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[blockIdx.x * 32 + (num_warp * 3) + warp_id] = k;
    __syncthreads();

    k = warp_id == 3 && lane_id < num_warp
            ? sum_cached_ptr[blockIdx.x * 32 + (num_warp * 3) + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 3 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            k += __shfl_down_sync(0x0ffffffff, k, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[blockIdx.x * 32 + 3] = k + bias3[relative_idx];
    }
}

template <unsigned int hidden_size>
__global__ void solve(const float *__restrict__ data,
                      const float *__restrict__ state_c,
                      float *__restrict__ output, float *__restrict__ new_state,
                      float *__restrict__ sum_cached_ptr) {
    // extern __shared__ float sum_cached[];

    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;

    if (threadIdx.x == 0) {
        float x = state_c[relative_idx] *
                      sigmoid(sum_cached_ptr[blockIdx.x * 32 + 2] + 1.0f) +
                  sigmoid(sum_cached_ptr[blockIdx.x * 32 + 0]) *
                      tanh(sum_cached_ptr[blockIdx.x * 32 + 1]);

        if (blockIdx.x < hidden_size)
            output[relative_idx] =
                tanh(x) * sigmoid(sum_cached_ptr[blockIdx.x * 32 + 3]);
        else
            new_state[relative_idx] = x;
    }
}

template __global__ void compute_x<256>(const float *__restrict__ data,
                                        const float *__restrict__ state_c,
                                        const float *__restrict__ state_h,
                                        const float *__restrict__ W0,
                                        const float *__restrict__ U0,
                                        const float *__restrict__ bias0,
                                        float *__restrict__ sum_cached_ptr);

template __global__ void compute_y<256>(const float *__restrict__ data,
                                        const float *__restrict__ state_c,
                                        const float *__restrict__ state_h,
                                        const float *__restrict__ W1,
                                        const float *__restrict__ U1,
                                        const float *__restrict__ bias1,
                                        float *__restrict__ sum_cached_ptr);

template __global__ void compute_z<256>(const float *__restrict__ data,
                                        const float *__restrict__ state_c,
                                        const float *__restrict__ state_h,
                                        const float *__restrict__ W2,
                                        const float *__restrict__ U2,
                                        const float *__restrict__ bias2,
                                        float *__restrict__ sum_cached_ptr);

template __global__ void compute_k<256>(const float *__restrict__ data,
                                        const float *__restrict__ state_c,
                                        const float *__restrict__ state_h,
                                        const float *__restrict__ W3,
                                        const float *__restrict__ U3,
                                        const float *__restrict__ bias3,
                                        float *__restrict__ sum_cached_ptr);

template __global__ void solve<256>(const float *__restrict__ data,
                                    const float *__restrict__ state_c,
                                    float *__restrict__ output,
                                    float *__restrict__ new_state,
                                    float *__restrict__ sum_cached_ptr);