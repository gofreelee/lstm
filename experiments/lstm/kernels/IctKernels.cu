#include "../net/wavefront_lstm_8_8_1_256/include/RammerLikeArgs.h"

static inline __device__ float sigmoid(float x) {
    return __fdividef(1.000000e+00f, 1.000000e+00f + __expf(0.000000e+00f - x));
}

#define castFloat4ToFloat(X) reinterpret_cast<float *>(const_cast<float4 *>(X))

static __device__ void matmul(dim3 blockIdx1, float *__restrict__ matrix,
                              float *__restrict__ vector,
                              float *__restrict__ output) {
    int warpIdx = threadIdx.x >> 5;
    int laneIdx = threadIdx.x & 31;
    int colOffset = (blockIdx1.x << 5) + laneIdx;

    float val = 0.0000f;
    int k_start = warpIdx * 64;
    int k_end = (warpIdx + 1) * 64;
#pragma unroll 64
    for (int i = k_start; i < k_end; ++i)
        val = fma(vector[i], matrix[i * 256 + colOffset], val);
    if (warpIdx == 0)
        output[colOffset] = 0.0000f;
    __syncthreads();
    atomicAdd(output + colOffset, val);
}

template <unsigned int t_hidden_size, bool update_state_c = true>
static inline __device__ void
point_to_point_func(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output) {

    float z = input->WMulDataResult[t_hidden_size * 2 + threadIdx.x] +
              input->UMulStateHResult[t_hidden_size * 2 + threadIdx.x] +
              model->bias[2][threadIdx.x];
    z = sigmoid(z + 1.0000f) * castFloat4ToFloat(input->state_c)[threadIdx.x];

    float x = input->WMulDataResult[threadIdx.x] +
              input->UMulStateHResult[threadIdx.x] +
              model->bias[0][threadIdx.x];
    x = sigmoid(x);

    float y = input->WMulDataResult[t_hidden_size + threadIdx.x] +
              input->UMulStateHResult[t_hidden_size + threadIdx.x] +
              model->bias[1][threadIdx.x];
    y = tanhf(y);

    float new_state_c = fma(x, y, z);
    if (update_state_c)
        castFloat4ToFloat(output->new_state_c)[threadIdx.x] = new_state_c;

    float w = input->WMulDataResult[t_hidden_size * 3 + threadIdx.x] +
              input->UMulStateHResult[t_hidden_size * 3 + threadIdx.x] +
              model->bias[3][threadIdx.x];
    w = sigmoid(w);
    castFloat4ToFloat(output->new_state_h)[threadIdx.x] =
        tanhf(new_state_c) * w;
}

template <unsigned int t_hidden_size, bool update_state_c, bool wd_computed,
          bool us_computed>
static inline __device__ void
onekernel_func(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
               RammerLikeCellModel<t_hidden_size> *__restrict__ model,
               RammerLikeCellOutput *__restrict__ output,
               float nndense_output[4][32], dim3 blockIdx1) {

    const int laneIdx = threadIdx.x & 0x1f;
    const int warpIdx = threadIdx.x >> 5;
    const int colOffset = (blockIdx1.x << 5) + laneIdx;
    float temp[4] = {0.0000f, 0.0000f, 0.0000f, 0.0000f};
    nndense_output[warpIdx][laneIdx] = 0.0000f;

    int vectorRow = warpIdx * 64;
    int kStart = vectorRow * t_hidden_size + colOffset;
    const int kEnd = kStart + 64 * t_hidden_size;
    for (; kStart < kEnd; kStart += t_hidden_size, ++vectorRow) {
        if (!wd_computed) {
            const float data = castFloat4ToFloat(input->data)[vectorRow];
#pragma unroll 4
            for (int i = 0; i < 4; ++i)
                temp[i] =
                    fma(castFloat4ToFloat(model->W[i])[kStart], data, temp[i]);
        }

        if (!us_computed) {
            const float stateh = castFloat4ToFloat(input->state_h)[vectorRow];
#pragma unroll 4
            for (int i = 0; i < 4; ++i)
                temp[i] = fma(castFloat4ToFloat(model->U[i])[kStart], stateh,
                              temp[i]);
        }
    }
    __syncthreads();

    if (warpIdx != 0) {
        atomicAdd(&nndense_output[0][laneIdx], temp[0]);
        atomicAdd(&nndense_output[1][laneIdx], temp[1]);
        atomicAdd(&nndense_output[2][laneIdx], temp[2]);
        atomicAdd(&nndense_output[3][laneIdx], temp[3]);
    } else {
        temp[0] += model->bias[0][colOffset];
        temp[1] += model->bias[1][colOffset];
        temp[2] += 1.0000f + model->bias[2][colOffset];
        temp[3] += model->bias[3][colOffset];

        if (wd_computed) {
#pragma unroll 4
            for (int i = 0; i < 4; ++i)
                temp[i] += input->WMulDataResult[i * t_hidden_size + colOffset];
        }

        if (us_computed) {
#pragma unroll 4
            for (int i = 0; i < 4; ++i)
                temp[i] +=
                    input->UMulStateHResult[i * t_hidden_size + colOffset];
        }
    }
    __syncthreads();

    if (warpIdx == 0) {
        float x = sigmoid(nndense_output[0][laneIdx] + temp[0]);
        float y = tanhf(nndense_output[1][laneIdx] + temp[1]);
        float z = sigmoid(nndense_output[2][laneIdx] + temp[2]) *
                  castFloat4ToFloat(input->state_c)[colOffset];
        float w = sigmoid(nndense_output[3][laneIdx] + temp[3]);
        float new_state_c = x * y + z;
        float new_state_h = tanhf(new_state_c) * w;
        if (update_state_c)
            castFloat4ToFloat(output->new_state_c)[colOffset] = new_state_c;
        castFloat4ToFloat(output->new_state_h)[colOffset] = new_state_h;
    }
}

#define WMulData(cell, step)                                                   \
    {                                                                          \
        matmul(blockIdx.x & 0x7, castFloat4ToFloat(models[cell].W[idx]),       \
               castFloat4ToFloat(inputs[step * t_num_layer + cell].data),      \
               &inputs[step * t_num_layer + cell]                              \
                    .WMulDataResult[idx * t_hidden_size]);                     \
    }

#define ok_hasu_update_c(cell, step)                                           \
    {                                                                          \
        onekernel_func<t_hidden_size, true, false, true>(                      \
            &inputs[step * t_num_layer + cell], &models[cell],                 \
            &outputs[step * t_num_layer + cell], nndense_output,               \
            blockIdx.x & 0x7);                                                 \
    }

#define ok_hasw_update_c(cell, step)                                           \
    {                                                                          \
        onekernel_func<t_hidden_size, true, true, false>(                      \
            &inputs[step * t_num_layer + cell], &models[cell],                 \
            &outputs[step * t_num_layer + cell], nndense_output,               \
            blockIdx.x & 0x7);                                                 \
    }

#define ok_hasw_not_update_c(cell, step)                                       \
    {                                                                          \
        onekernel_func<t_hidden_size, false, true, false>(                     \
            &inputs[step * t_num_layer + cell], &models[cell],                 \
            &outputs[step * t_num_layer + cell], nndense_output,               \
            blockIdx.x & 0x7);                                                 \
    }

#define ok_update_c(cell, step)                                                \
    {                                                                          \
        onekernel_func<t_hidden_size, true, false, false>(                     \
            &inputs[step * t_num_layer + cell], &models[cell],                 \
            &outputs[step * t_num_layer + cell], nndense_output,               \
            blockIdx.x & 0x7);                                                 \
    }

#define ok_not_update_c(cell, step)                                            \
    {                                                                          \
        onekernel_func<t_hidden_size, false, false, false>(                    \
            &inputs[step * t_num_layer + cell], &models[cell],                 \
            &outputs[step * t_num_layer + cell], nndense_output,               \
            blockIdx.x & 0x7);                                                 \
    }

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok_1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {

    const int step = blockIdx.x >> 5;
    const int idx = (blockIdx.x & 0x1f) >> 3;
    WMulData(0, step);
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(256, 1)
    ok0(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {

    point_to_point_func(&inputs[0], &models[0], &outputs[0]);
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 1);
        break;
    case 1:
        ok_hasu_update_c(1, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok2(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 2);
        break;
    case 1:
        ok_update_c(1, 1);
        break;
    case 2:
        ok_hasu_update_c(2, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok3(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 3);
        break;
    case 1:
        ok_update_c(1, 2);
        break;
    case 2:
        ok_update_c(2, 1);
        break;
    case 3:
        ok_hasu_update_c(3, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok4(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 4);
        break;
    case 1:
        ok_update_c(1, 3);
        break;
    case 2:
        ok_update_c(2, 2);
        break;
    case 3:
        ok_update_c(3, 1);
        break;
    case 4:
        ok_hasu_update_c(4, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok5(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 5);
        break;
    case 1:
        ok_update_c(1, 4);
        break;
    case 2:
        ok_update_c(2, 3);
        break;
    case 3:
        ok_update_c(3, 2);
        break;
    case 4:
        ok_update_c(4, 1);
        break;
    case 5:
        ok_hasu_update_c(5, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok6(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 6);
        break;
    case 1:
        ok_update_c(1, 5);
        break;
    case 2:
        ok_update_c(2, 4);
        break;
    case 3:
        ok_update_c(3, 3);
        break;
    case 4:
        ok_update_c(4, 2);
        break;
    case 5:
        ok_update_c(5, 1);
        break;
    case 6:
        ok_hasu_update_c(6, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok7(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_not_update_c(0, 7);
        break;
    case 1:
        ok_update_c(1, 6);
        break;
    case 2:
        ok_update_c(2, 5);
        break;
    case 3:
        ok_update_c(3, 4);
        break;
    case 4:
        ok_update_c(4, 3);
        break;
    case 5:
        ok_update_c(5, 2);
        break;
    case 6:
        ok_update_c(6, 1);
        break;
    case 7:
        ok_hasu_update_c(7, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok8(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(1, 7);
        break;
    case 1:
        ok_update_c(2, 6);
        break;
    case 2:
        ok_update_c(3, 5);
        break;
    case 3:
        ok_update_c(4, 4);
        break;
    case 4:
        ok_update_c(5, 3);
        break;
    case 5:
        ok_update_c(6, 2);
        break;
    case 6:
        ok_update_c(7, 1);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok9(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(2, 7);
        break;
    case 1:
        ok_update_c(3, 6);
        break;
    case 2:
        ok_update_c(4, 5);
        break;
    case 3:
        ok_update_c(5, 4);
        break;
    case 4:
        ok_update_c(6, 3);
        break;
    case 5:
        ok_update_c(7, 2);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok10(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(3, 7);
        break;
    case 1:
        ok_update_c(4, 6);
        break;
    case 2:
        ok_update_c(5, 5);
        break;
    case 3:
        ok_update_c(6, 4);
        break;
    case 4:
        ok_update_c(7, 3);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok11(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(4, 7);
        break;
    case 1:
        ok_update_c(5, 6);
        break;
    case 2:
        ok_update_c(6, 5);
        break;
    case 3:
        ok_update_c(7, 4);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok12(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(5, 7);
        break;
    case 1:
        ok_update_c(6, 6);
        break;
    case 2:
        ok_update_c(7, 5);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok13(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(6, 7);
        break;
    case 1:
        ok_update_c(7, 6);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok14(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    ok_not_update_c(7, 7);
}

template __global__ void
    ok_1<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok0<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok1<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok2<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok3<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok4<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok5<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok6<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok7<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok8<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok9<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                RammerLikeCellModel<256> *__restrict__ models,
                RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok10<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok11<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok12<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok13<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok14<256, 8>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);