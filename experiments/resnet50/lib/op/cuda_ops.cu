#include "cuda_ops.h"

template <int kSize, bool kFuseRelu>
__global__ void operator_vecaddvec_h(const Tensor<kSize> *__restrict__ input1,
                                     const Tensor<kSize> *__restrict__ input2,
                                     Tensor<kSize> *__restrict__ output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < kSize) {
        float value = input1->data[tid] + input2->data[tid];
        if constexpr (kFuseRelu) {
            output->data[tid] = value > 0 ? value : 0;
        } else {
            output->data[tid] = value;
        }
    }
}

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, bool kFuseRelu>
__global__ void operator_fuse_matmul_bias_relu_h(
    const Tensor<kBatch1 * kHeight * kK> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2> *__restrict__ output,
    const Tensor<kBatch1 * kHeight * kWidth * kBatch2> *__restrict__ bias) {
    __shared__ float shared_input1[kTileSize][kTileSize];
    __shared__ float shared_input2[kTileSize][kTileSize];

    int batch_idx = blockIdx.z;
    int input1_off = 0;
    int input2_off = 0;
    int output_off = 0;
    if (kBroadCast != 1)
        input1_off = batch_idx * kHeight * kK;
    if (kBroadCast != 2)
        input2_off = batch_idx * kK * kWidth;

    output_off = batch_idx * kHeight * kWidth;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * kTileSize + tx;
    int col = by * kTileSize + ty;
    float v = 0;
#pragma unroll 8
    for (int i = 0; i < (kK + kTileSize - 1) / kTileSize; i++) {
        if (i * kTileSize + ty < kK && row < kHeight)
            shared_input1[tx][ty] =
                input1->data[input1_off + row * kK + i * kTileSize + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * kTileSize + tx < kK && col < kWidth)
            shared_input2[tx][ty] =
                input2->data[input2_off + (i * kTileSize + tx) * kWidth + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();
#pragma unroll 16
        for (int j = 0; j < kTileSize; j++)
            v += shared_input1[tx][j] * shared_input2[j][ty];
        __syncthreads();
    }

    if (row < kHeight && col < kWidth) {
        v += bias->data[output_off + row * kWidth + col];
        if constexpr (kFuseRelu) {
            v = v > 0 ? v : 0;
        }
        output->data[output_off + row * kWidth + col] = v;
    }
}

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, bool kFuseRelu>
__global__ void operator_fuse_matmul_relu_h(
    const Tensor<kBatch1 * kHeight * kK> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2> *__restrict__ output) {
    __shared__ float shared_input1[kTileSize][kTileSize];
    __shared__ float shared_input2[kTileSize][kTileSize];

    int batch_idx = blockIdx.z;
    int input1_off = 0;
    int input2_off = 0;
    int output_off = 0;
    if (kBroadCast != 1)
        input1_off = batch_idx * kHeight * kK;
    if (kBroadCast != 2)
        input2_off = batch_idx * kK * kWidth;

    output_off = batch_idx * kHeight * kWidth;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * kTileSize + tx;
    int col = by * kTileSize + ty;
    float v = 0;
#pragma unroll 64
    for (int i = 0; i < (kK + kTileSize - 1) / kTileSize; i++) {
        if (i * kTileSize + ty < kK && row < kHeight)
            shared_input1[tx][ty] =
                input1->data[input1_off + row * kK + i * kTileSize + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * kTileSize + tx < kK && col < kWidth)
            shared_input2[tx][ty] =
                input2->data[input2_off + (i * kTileSize + tx) * kWidth + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();
#pragma unroll 16
        for (int j = 0; j < kTileSize; j++)
            v += shared_input1[tx][j] * shared_input2[j][ty];
        __syncthreads();
    }

    if (row < kHeight && col < kWidth) {
        if constexpr (kFuseRelu) {
            v = v > 0 ? v : 0;
        }
        output->data[output_off + row * kWidth + col] = v;
    }
}

template <int kBatchSize, int kChannels, int kHeight, int kWidth, int kKernelH,
          int kKernelW, int kPadH, int kPadW, int kStrideH, int kStrideW,
          int kInput, int kSize>
__global__ void operator_avg_pool_h(const Tensor<kInput> *__restrict__ input,
                                    AvgPoolState<kSize> *__restrict__ state) {

    int pooled_height = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
    int pooled_width = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
    int nthreads = kBatchSize * kChannels * pooled_height * pooled_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads) {
        // output location
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % kChannels;
        const int n = index / pooled_width / pooled_height / kChannels;

        // pooled range
        int hstart = ph * kStrideH - kPadH;
        int wstart = pw * kStrideW - kPadW;
        const int hend = fminf(hstart + kKernelH, kHeight);
        const int wend = fminf(wstart + kKernelW, kWidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        float avgval = 0.0f;
        int maxidx = -1;
        int slice_offset = (n * kChannels + c) * kHeight * kWidth;
#pragma unroll 4
        for (int h = hstart; h < hend; ++h) {
#pragma unroll 4
            for (int w = wstart; w < wend; ++w) {
                avgval = (input->data[slice_offset + h * kWidth + w]) /
                             ((hend - hstart) * (wend - wstart)) +
                         avgval;
            }
        }
        maxidx = hstart * kWidth +
                 wstart; // maxidx record left up corner position //
                         // data before pooling to get output data

        // output
        state->output.data[index] = avgval;

        // record idx
        state->mask.data[index] = maxidx;
    }
}

template <int kBatchSize, int kChannel, int kHeight, int kWidth>
static inline __device__ float
batch_normalization_func(float input,
                         const BatchNormParam<kChannel> *__restrict__ bn_params,
                         int index) {
    int channel_size = kHeight * kWidth;
    int feature_size = kChannel * kHeight * kWidth;

    int batch = index / feature_size;
    int channel = (index - batch * kChannel * kHeight * kWidth) / channel_size;

    float inv_var = bn_params->running_var.data[channel];
    inv_var = sqrtf(inv_var + 1e-5);
    inv_var = 1 / inv_var;
    float weight_v = bn_params->weight.data[channel];
    float bias_v = bn_params->bias.data[channel];
    float alpha = inv_var * weight_v;
    float mean_data = bn_params->running_mean.data[channel];
    float beta = bias_v - mean_data * alpha;
    return input * alpha + beta;
}

template <int kBatchSize, int kChannel, int kHeight, int kWidth, int kSize>
__global__ void operator_batch_normalization_h(
    const Tensor<kSize> *__restrict__ input,
    BatchNormState<kSize, kChannel> *__restrict__ state) {
    int total_size = kBatchSize * kChannel * kHeight * kWidth;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < total_size) {
        state->output.data[tid] =
            batch_normalization_func<kBatchSize, kChannel, kHeight, kWidth>(
                input->data[tid], &state->param, tid);
    }
}

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, int kConvChannel, int kConvHeight, int kConvWidth,
          bool kFuseRelu>
__global__ void operator_fuse_conv_bn_relu_h(
    const Tensor<kBatch1 * kHeight * kK> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2> *__restrict__ output,
    const BatchNormParam<kConvChannel> *__restrict__ bn_param) {
    static_assert(kConvChannel * kConvHeight * kConvWidth == kWidth * kHeight);
    __shared__ float shared_input1[kTileSize][kTileSize];
    __shared__ float shared_input2[kTileSize][kTileSize];

    int batch_idx = blockIdx.z;
    int input1_off = 0;
    int input2_off = 0;
    int output_off = 0;
    if (kBroadCast != 1)
        input1_off = batch_idx * kHeight * kK;
    if (kBroadCast != 2)
        input2_off = batch_idx * kK * kWidth;

    output_off = batch_idx * kHeight * kWidth;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * kTileSize + tx;
    int col = by * kTileSize + ty;
    float v = 0;
#pragma unroll 64
    for (int i = 0; i < (kK + kTileSize - 1) / kTileSize; i++) {
        if (i * kTileSize + ty < kK && row < kHeight)
            shared_input1[tx][ty] =
                input1->data[input1_off + row * kK + i * kTileSize + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * kTileSize + tx < kK && col < kWidth)
            shared_input2[tx][ty] =
                input2->data[input2_off + (i * kTileSize + tx) * kWidth + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();
#pragma unroll 16
        for (int j = 0; j < kTileSize; j++)
            v += shared_input1[tx][j] * shared_input2[j][ty];
        __syncthreads();
    }

    if (row < kHeight && col < kWidth) {
        int index = output_off + row * kWidth + col;
        v = batch_normalization_func<kBatch2, kConvChannel, kConvHeight,
                                     kConvWidth>(v, bn_param, index);
        if constexpr (kFuseRelu) {
            output->data[index] = v > 0 ? v : 0;
        } else {
            output->data[index] = v;
        }
    }
}

template <int kHeight, int kK, int kWidth, int kBroadCast, int kBatch1,
          int kBatch2, bool kFuseRelu>
__global__ void operator_fuse_conv_bias_relu_h(
    const Tensor<kBatch1 * kHeight * kK> *__restrict__ input1,
    const Tensor<kBatch2 * kWidth * kK> *__restrict__ input2,
    Tensor<kBatch1 * kHeight * kWidth * kBatch2> *__restrict__ output,
    const Tensor<kHeight> *__restrict__ bias) {

    __shared__ float shared_input1[kTileSize][kTileSize];
    __shared__ float shared_input2[kTileSize][kTileSize];

    int batch_idx = blockIdx.z;
    int input1_off = 0;
    int input2_off = 0;
    int output_off = 0;
    if (kBroadCast != 1)
        input1_off = batch_idx * kHeight * kK;
    if (kBroadCast != 2)
        input2_off = batch_idx * kK * kWidth;

    output_off = batch_idx * kHeight * kWidth;

    int bx = blockIdx.y;
    int by = blockIdx.x;
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int row = bx * kTileSize + tx;
    int col = by * kTileSize + ty;
    float v = 0;
#pragma unroll 8
    for (int i = 0; i < (kK + kTileSize - 1) / kTileSize; i++) {
        if (i * kTileSize + ty < kK && row < kHeight)
            shared_input1[tx][ty] =
                input1->data[input1_off + row * kK + i * kTileSize + ty];
        else
            shared_input1[tx][ty] = 0;

        if (i * kTileSize + tx < kK && col < kWidth)
            shared_input2[tx][ty] =
                input2->data[input2_off + (i * kTileSize + tx) * kWidth + col];
        else
            shared_input2[tx][ty] = 0;
        __syncthreads();
#pragma unroll 16
        for (int j = 0; j < kTileSize; j++)
            v += shared_input1[tx][j] * shared_input2[j][ty];
        __syncthreads();
    }

    if (row < kHeight && col < kWidth) {
        v += bias->data[row];
        if constexpr (kFuseRelu) {
            v = v > 0 ? v : 0;
        }
        output->data[output_off + row * kWidth + col] = v;
    }
}

template <int kBatchSize, int kHeight, int kWidth, int kChannelIn,
          int kChannelOut, int kKernelH, int kKernelW, int kPadH, int kPadW,
          int kStrideH, int kStrideW, bool kIsBias, int kInputSize,
          int kColSize>
__global__ void im2col_h(const Tensor<kInputSize> *__restrict__ input,
                         Tensor<kColSize> *__restrict__ col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int height_col = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
    int width_col = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
    int n = kChannelIn * height_col * width_col;
    int im_stride = kChannelIn * kHeight * kWidth;
    int col_stride = kChannelIn * kKernelH * kKernelW * height_col * width_col;

    if (index < n) {
        int batch_idx = blockIdx.y;
        int input_offset = batch_idx * im_stride;
        int col_offset = batch_idx * col_stride;

        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kKernelH * kKernelW;
        const int h_offset = h_col * kStrideH - kPadH;
        const int w_offset = w_col * kStrideW - kPadW;

        // channel offset
        col_offset += (c_col * height_col + h_col) * width_col + w_col;
        input_offset += (c_im * kHeight + h_offset) * kWidth + w_offset;

// copy to col
#pragma unroll 4
        for (int i = 0; i < kKernelH; ++i) {
#pragma unroll 4
            for (int j = 0; j < kKernelW; ++j) {
                int h_im = h_offset + i;
                int w_im = w_offset + j;
                col->data[col_offset] =
                    (h_im >= 0 && w_im >= 0 && h_im < kHeight && w_im < kWidth)
                        ? input->data[input_offset + i * kWidth + j]
                        : 0;
                col_offset += height_col * width_col;
            }
        }
    }
}

template <int kBatchSize, int kChannels, int kHeight, int kWidth, int kKernelH,
          int kKernelW, int kPadH, int kPadW, int kStrideH, int kStrideW,
          int kInput, int kSize>
__global__ void operator_max_pool_h(const Tensor<kInput> *__restrict__ input,
                                    AvgPoolState<kSize> *__restrict__ state) {
    int pooled_height = (kHeight + 2 * kPadH - kKernelH) / kStrideH + 1;
    int pooled_width = (kWidth + 2 * kPadW - kKernelW) / kStrideW + 1;
    int nthreads = kBatchSize * kChannels * pooled_height * pooled_width;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nthreads) {
        // output location
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / pooled_width / pooled_height) % kChannels;
        const int n = index / pooled_width / pooled_height / kChannels;

        // pooled range
        int hstart = ph * kStrideH - kPadH;
        int wstart = pw * kStrideW - kPadW;
        const int hend = fminf(hstart + kKernelH, kHeight);
        const int wend = fminf(wstart + kKernelW, kWidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        // get max value postion
        float maxval = -FLT_MAX;
        int maxidx = -1;
        int slice_offset = (n * kChannels + c) * kHeight * kWidth;

#pragma unroll 4
        for (int h = hstart; h < hend; ++h) {
#pragma unroll 4
            for (int w = wstart; w < wend; ++w) {
                if (input->data[slice_offset + h * kWidth + w] > maxval) {
                    maxidx = h * kWidth + w;
                    maxval = input->data[slice_offset + maxidx];
                }
            }
        }
        // output
        state->output.data[index] = maxval;

        // record idx
        state->mask.data[index] = maxidx;
    }
}

template <int kSize>
__global__ void operator_vectorrelu_h(const Tensor<kSize> *input,
                                      Tensor<kSize> *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < kSize) {
        output->data[tid] = input->data[tid] > 0 ? input->data[tid] : 0;
    }
}

template __global__ void
    im2col_h<1, 224, 224, 3, 64, 7, 7, 3, 3, 2, 2, false, 150528, 1843968>(
        Tensor<150528> const *, Tensor<1843968> *);

template __global__ void
operator_fuse_conv_bn_relu_h<64, 147, 12544, 1, 1, 1, 64, 112, 112, true>(
    Tensor<((1) * (64)) * (147)> const *,
    Tensor<((1) * (12544)) * (147)> const *,
    Tensor<(((1) * (64)) * (12544)) * (1)> *, BatchNormParam<64> const *);
template __global__ void
    operator_max_pool_h<1, 64, 112, 112, 3, 3, 1, 1, 2, 2, 802816, 200704>(
        Tensor<802816> const *, AvgPoolState<200704> *);
template __global__ void
    operator_avg_pool_h<1, 2048, 7, 7, 7, 7, 0, 0, 7, 7, 100352, 2048>(
        Tensor<100352> const *, AvgPoolState<2048> *);
template __global__ void
operator_fuse_matmul_bias_relu_h<1, 2048, 1000, 0, 1, 1, false>(
    Tensor<((1) * (1)) * (2048)> const *,
    Tensor<((1) * (1000)) * (2048)> const *,
    Tensor<(((1) * (1)) * (1000)) * (1)> *,
    Tensor<(((1) * (1)) * (1000)) * (1)> const *);
template __global__ void
    im2col_h<1, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1, false, 200704, 200704>(
        Tensor<200704> const *, Tensor<200704> *);
template __global__ void
operator_fuse_conv_bn_relu_h<64, 64, 3136, 1, 1, 1, 64, 56, 56, true>(
    Tensor<((1) * (64)) * (64)> const *, Tensor<((1) * (3136)) * (64)> const *,
    Tensor<(((1) * (64)) * (3136)) * (1)> *, BatchNormParam<64> const *);
template __global__ void
    im2col_h<1, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, false, 200704, 1806336>(
        Tensor<200704> const *, Tensor<1806336> *);
template __global__ void
operator_fuse_conv_bn_relu_h<64, 576, 3136, 1, 1, 1, 64, 56, 56, true>(
    Tensor<((1) * (64)) * (576)> const *,
    Tensor<((1) * (3136)) * (576)> const *,
    Tensor<(((1) * (64)) * (3136)) * (1)> *, BatchNormParam<64> const *);
template __global__ void
    im2col_h<1, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1, false, 200704, 200704>(
        Tensor<200704> const *, Tensor<200704> *);
template __global__ void
operator_fuse_conv_bn_relu_h<256, 64, 3136, 1, 1, 1, 256, 56, 56, false>(
    Tensor<((1) * (256)) * (64)> const *, Tensor<((1) * (3136)) * (64)> const *,
    Tensor<(((1) * (256)) * (3136)) * (1)> *, BatchNormParam<256> const *);
template __global__ void operator_vecaddvec_h<802816, true>(
    Tensor<802816> const *, Tensor<802816> const *, Tensor<802816> *);
template __global__ void
    im2col_h<1, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1, false, 802816, 802816>(
        Tensor<802816> const *, Tensor<802816> *);
template __global__ void
operator_fuse_conv_bn_relu_h<64, 256, 3136, 1, 1, 1, 64, 56, 56, true>(
    Tensor<((1) * (64)) * (256)> const *,
    Tensor<((1) * (3136)) * (256)> const *,
    Tensor<(((1) * (64)) * (3136)) * (1)> *, BatchNormParam<64> const *);
template __global__ void
    im2col_h<1, 56, 56, 256, 128, 1, 1, 0, 0, 1, 1, false, 802816, 802816>(
        Tensor<802816> const *, Tensor<802816> *);
template __global__ void
operator_fuse_conv_bn_relu_h<128, 256, 3136, 1, 1, 1, 128, 56, 56, true>(
    Tensor<((1) * (128)) * (256)> const *,
    Tensor<((1) * (3136)) * (256)> const *,
    Tensor<(((1) * (128)) * (3136)) * (1)> *, BatchNormParam<128> const *);
template __global__ void
    im2col_h<1, 56, 56, 128, 128, 3, 3, 1, 1, 2, 2, false, 401408, 903168>(
        Tensor<401408> const *, Tensor<903168> *);
template __global__ void
    im2col_h<1, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1, false, 100352, 100352>(
        Tensor<100352> const *, Tensor<100352> *);
template __global__ void
operator_fuse_conv_bn_relu_h<512, 128, 784, 1, 1, 1, 512, 28, 28, false>(
    Tensor<((1) * (512)) * (128)> const *,
    Tensor<((1) * (784)) * (128)> const *,
    Tensor<(((1) * (512)) * (784)) * (1)> *, BatchNormParam<512> const *);
template __global__ void
    im2col_h<1, 56, 56, 256, 512, 1, 1, 0, 0, 2, 2, false, 802816, 200704>(
        Tensor<802816> const *, Tensor<200704> *);
template __global__ void
operator_fuse_conv_bn_relu_h<512, 256, 784, 1, 1, 1, 512, 28, 28, false>(
    Tensor<((1) * (512)) * (256)> const *,
    Tensor<((1) * (784)) * (256)> const *,
    Tensor<(((1) * (512)) * (784)) * (1)> *, BatchNormParam<512> const *);
template __global__ void operator_vecaddvec_h<401408, true>(
    Tensor<401408> const *, Tensor<401408> const *, Tensor<401408> *);
template __global__ void
    im2col_h<1, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1, false, 401408, 401408>(
        Tensor<401408> const *, Tensor<401408> *);
template __global__ void
operator_fuse_conv_bn_relu_h<128, 512, 784, 1, 1, 1, 128, 28, 28, true>(
    Tensor<((1) * (128)) * (512)> const *,
    Tensor<((1) * (784)) * (512)> const *,
    Tensor<(((1) * (128)) * (784)) * (1)> *, BatchNormParam<128> const *);
template __global__ void
    im2col_h<1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, false, 100352, 903168>(
        Tensor<100352> const *, Tensor<903168> *);
template __global__ void
operator_fuse_conv_bn_relu_h<128, 1152, 784, 1, 1, 1, 128, 28, 28, true>(
    Tensor<((1) * (128)) * (1152)> const *,
    Tensor<((1) * (784)) * (1152)> const *,
    Tensor<(((1) * (128)) * (784)) * (1)> *, BatchNormParam<128> const *);
template __global__ void
    im2col_h<1, 28, 28, 512, 256, 1, 1, 0, 0, 1, 1, false, 401408, 401408>(
        Tensor<401408> const *, Tensor<401408> *);
template __global__ void
operator_fuse_conv_bn_relu_h<256, 512, 784, 1, 1, 1, 256, 28, 28, true>(
    Tensor<((1) * (256)) * (512)> const *,
    Tensor<((1) * (784)) * (512)> const *,
    Tensor<(((1) * (256)) * (784)) * (1)> *, BatchNormParam<256> const *);
template __global__ void
    im2col_h<1, 28, 28, 256, 256, 3, 3, 1, 1, 2, 2, false, 200704, 451584>(
        Tensor<200704> const *, Tensor<451584> *);
template __global__ void
    im2col_h<1, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1, false, 50176, 50176>(
        Tensor<50176> const *, Tensor<50176> *);
template __global__ void
operator_fuse_conv_bn_relu_h<1024, 256, 196, 1, 1, 1, 1024, 14, 14, false>(
    Tensor<((1) * (1024)) * (256)> const *,
    Tensor<((1) * (196)) * (256)> const *,
    Tensor<(((1) * (1024)) * (196)) * (1)> *, BatchNormParam<1024> const *);
template __global__ void
    im2col_h<1, 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2, false, 401408, 100352>(
        Tensor<401408> const *, Tensor<100352> *);
template __global__ void
operator_fuse_conv_bn_relu_h<1024, 512, 196, 1, 1, 1, 1024, 14, 14, false>(
    Tensor<((1) * (1024)) * (512)> const *,
    Tensor<((1) * (196)) * (512)> const *,
    Tensor<(((1) * (1024)) * (196)) * (1)> *, BatchNormParam<1024> const *);
template __global__ void operator_vecaddvec_h<200704, true>(
    Tensor<200704> const *, Tensor<200704> const *, Tensor<200704> *);
template __global__ void
    im2col_h<1, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1, false, 200704, 200704>(
        Tensor<200704> const *, Tensor<200704> *);
template __global__ void
operator_fuse_conv_bn_relu_h<256, 1024, 196, 1, 1, 1, 256, 14, 14, true>(
    Tensor<((1) * (256)) * (1024)> const *,
    Tensor<((1) * (196)) * (1024)> const *,
    Tensor<(((1) * (256)) * (196)) * (1)> *, BatchNormParam<256> const *);
template __global__ void
    im2col_h<1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, false, 50176, 451584>(
        Tensor<50176> const *, Tensor<451584> *);
template __global__ void
operator_fuse_conv_bn_relu_h<256, 2304, 196, 1, 1, 1, 256, 14, 14, true>(
    Tensor<((1) * (256)) * (2304)> const *,
    Tensor<((1) * (196)) * (2304)> const *,
    Tensor<(((1) * (256)) * (196)) * (1)> *, BatchNormParam<256> const *);
template __global__ void
    im2col_h<1, 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1, false, 200704, 200704>(
        Tensor<200704> const *, Tensor<200704> *);
template __global__ void
operator_fuse_conv_bn_relu_h<512, 1024, 196, 1, 1, 1, 512, 14, 14, true>(
    Tensor<((1) * (512)) * (1024)> const *,
    Tensor<((1) * (196)) * (1024)> const *,
    Tensor<(((1) * (512)) * (196)) * (1)> *, BatchNormParam<512> const *);
template __global__ void
    im2col_h<1, 14, 14, 512, 512, 3, 3, 1, 1, 2, 2, false, 100352, 225792>(
        Tensor<100352> const *, Tensor<225792> *);
template __global__ void
    im2col_h<1, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1, false, 25088, 25088>(
        Tensor<25088> const *, Tensor<25088> *);
template __global__ void
operator_fuse_conv_bn_relu_h<2048, 512, 49, 1, 1, 1, 2048, 7, 7, false>(
    Tensor<((1) * (2048)) * (512)> const *,
    Tensor<((1) * (49)) * (512)> const *,
    Tensor<(((1) * (2048)) * (49)) * (1)> *, BatchNormParam<2048> const *);
template __global__ void
    im2col_h<1, 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2, false, 200704, 50176>(
        Tensor<200704> const *, Tensor<50176> *);
template __global__ void
operator_fuse_conv_bn_relu_h<2048, 1024, 49, 1, 1, 1, 2048, 7, 7, false>(
    Tensor<((1) * (2048)) * (1024)> const *,
    Tensor<((1) * (49)) * (1024)> const *,
    Tensor<(((1) * (2048)) * (49)) * (1)> *, BatchNormParam<2048> const *);
template __global__ void operator_vecaddvec_h<100352, true>(
    Tensor<100352> const *, Tensor<100352> const *, Tensor<100352> *);
template __global__ void
    im2col_h<1, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1, false, 100352, 100352>(
        Tensor<100352> const *, Tensor<100352> *);
template __global__ void
operator_fuse_conv_bn_relu_h<512, 2048, 49, 1, 1, 1, 512, 7, 7, true>(
    Tensor<((1) * (512)) * (2048)> const *,
    Tensor<((1) * (49)) * (2048)> const *,
    Tensor<(((1) * (512)) * (49)) * (1)> *, BatchNormParam<512> const *);
template __global__ void
    im2col_h<1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, false, 25088, 225792>(
        Tensor<25088> const *, Tensor<225792> *);
template __global__ void
operator_fuse_conv_bn_relu_h<512, 4608, 49, 1, 1, 1, 512, 7, 7, true>(
    Tensor<((1) * (512)) * (4608)> const *,
    Tensor<((1) * (49)) * (4608)> const *,
    Tensor<(((1) * (512)) * (49)) * (1)> *, BatchNormParam<512> const *);