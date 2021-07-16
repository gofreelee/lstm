__device__ static inline float sigmoid(float x) {
    return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}
#include <stdio.h>
// blockDim.x <= 256
template <unsigned int hidden_size>
__global__ void
OneKernel(const float *__restrict__ data, const float *__restrict__ state_c,
          const float *__restrict__ state_h, const float *__restrict__ W0,
          const float *__restrict__ W1, const float *__restrict__ W2,
          const float *__restrict__ W3, const float *__restrict__ U0,
          const float *__restrict__ U1, const float *__restrict__ U2,
          const float *__restrict__ U3, const float *__restrict__ bias0,
          const float *__restrict__ bias1, const float *__restrict__ bias2,
          const float *__restrict__ bias3, float *__restrict__ output,
          float *__restrict__ new_state) {

    extern __shared__ float sum_cached[];
    float *sum_cached_ptr = (float *)sum_cached;

    float x, y, z, k;
    int num_warp = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1f;
    int relative_idx =
        (blockIdx.x >= hidden_size) ? blockIdx.x - hidden_size : blockIdx.x;
    float private_vector_element0 = data[threadIdx.x],
          private_vector_element1 = state_h[threadIdx.x];

    x = W0[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    x += U0[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        x += __shfl_down_sync(0xffffffff, x, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[warp_id] = x;

    y = W1[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    y += U1[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        y += __shfl_down_sync(0xffffffff, y, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[num_warp + warp_id] = y;

    z = W2[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    z += U2[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        z += __shfl_down_sync(0xffffffff, z, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[(num_warp << 1) + warp_id] = z;

    k = W3[threadIdx.x * hidden_size + relative_idx] * private_vector_element0;
    k += U3[threadIdx.x * hidden_size + relative_idx] * private_vector_element1;
#pragma unroll 5
    for (int i = 16; i > 0; i >>= 1)
        k += __shfl_down_sync(0xffffffff, k, i, 32);
    if (lane_id == 0)
        sum_cached_ptr[num_warp * 3 + warp_id] = k;
    __syncthreads();

    // in the following code segment, warp_id may be wrong if hidden_size < 256
    x = warp_id == 0 && lane_id < num_warp ? sum_cached_ptr[lane_id] : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 0 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            x += __shfl_down_sync(0x0ff, x, i, 32); // FIX
        if (lane_id == 0) {
            sum_cached_ptr[0] = x + bias0[relative_idx];
        }
    }

    y = warp_id == 1 && lane_id < num_warp ? sum_cached_ptr[num_warp + lane_id]
                                           : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 1 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            y += __shfl_down_sync(0x0ff, y, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[1] = y + bias1[relative_idx];
    }

    z = warp_id == 2 && lane_id < num_warp
            ? sum_cached_ptr[(num_warp << 1) + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 2 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            z += __shfl_down_sync(0x0ff, z, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[2] = z + bias2[relative_idx];
    }

    k = warp_id == 3 && lane_id < num_warp
            ? sum_cached_ptr[(num_warp * 3) + lane_id]
            : 0.0f;
    __syncwarp(0xffffffff);
    if (warp_id == 3 && lane_id < num_warp) {
#pragma unroll 3
        for (int i = 4; i > 0; i >>= 1)
            k += __shfl_down_sync(0x0ff, k, i, 32);
        if (lane_id == 0)
            sum_cached_ptr[3] = k + bias3[relative_idx];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        x = state_c[relative_idx] * sigmoid(sum_cached_ptr[2] + 1.0f) +
            sigmoid(sum_cached_ptr[0]) * tanh(sum_cached_ptr[1]);

        if (blockIdx.x < hidden_size)
            output[relative_idx] = tanh(x) * sigmoid(sum_cached_ptr[3]);
        else
            new_state[relative_idx] = x;
    }
}

template __global__ void
OneKernel<256>(const float *__restrict__ data,
               const float *__restrict__ state_c,
               const float *__restrict__ state_h, const float *__restrict__ W0,
               const float *__restrict__ W1, const float *__restrict__ W2,
               const float *__restrict__ W3, const float *__restrict__ U0,
               const float *__restrict__ U1, const float *__restrict__ U2,
               const float *__restrict__ U3, const float *__restrict__ bias0,
               const float *__restrict__ bias1, const float *__restrict__ bias2,
               const float *__restrict__ bias3, float *__restrict__ output,
               float *__restrict__ new_state);