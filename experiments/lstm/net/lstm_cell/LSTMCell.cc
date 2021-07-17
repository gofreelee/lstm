#include "LSTMCell.h"

#define COLUMNS_PER_BLOCK 32
void gemv(const float *__restrict__ input, const float *__restrict__ weight,
          float *__restrict__ output);
void solve(float *t00, float *t01, float *b0, float *t10, float *t11, float *b1,
           float *t20, float *t21, float *b2, float *t30, float *t31, float *b3,
           float *state_c_old, float *state_c, float *state_h);
namespace mica::experiments::lstm {

void LSTMCell::init(const HostCellParams &params) {
    cudaMalloc(&input_dev, sizeof(float) * (8 * hidden_size * input_size +
                                            8 * hidden_size + input_size));
    output_host = (float *)malloc(sizeof(float) * hidden_size);

    state_c_dev = input_dev + input_size;
    state_c_new_dev = state_c_dev + hidden_size;
    state_h_dev = state_c_new_dev + hidden_size;
    output_dev = state_h_dev + hidden_size;
    for (int i = 0; i < 4; ++i) {
        W_dev[i] = output_dev + i * hidden_size * input_size;
        U_dev[i] = output_dev + (i + 4) * hidden_size * input_size;
        bias_dev[i] =
            output_dev + 8 * hidden_size * input_size + i * hidden_size;
    }
    cudaMemcpy(input_dev, params.input, sizeof(float) * input_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_c_dev, params.init_state_c, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_h_dev, params.init_state_h, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(W_dev[0], params.W, sizeof(float) * hidden_size * input_size * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(U_dev[0], params.U, sizeof(float) * hidden_size * input_size * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_dev, params.bias, sizeof(float) * hidden_size * 4,
               cudaMemcpyHostToDevice);
    cudaStreamCreate(&stream_t);
}

void LSTMCell::compute() {
    float *tmp_outputs[8];
    for (int i = 0; i < 8; ++i)
        cudaMalloc(&tmp_outputs[i], sizeof(float) * hidden_size);
    void *WI_0[] = {&input_dev, &W_dev[0], &tmp_outputs[0]};
    void *WI_1[] = {&input_dev, &W_dev[1], &tmp_outputs[1]};
    void *WI_2[] = {&input_dev, &W_dev[2], &tmp_outputs[2]};
    void *WI_3[] = {&input_dev, &W_dev[3], &tmp_outputs[3]};
    void *UH_0[] = {&state_h_dev, &U_dev[0], &tmp_outputs[4]};
    void *UH_1[] = {&state_h_dev, &U_dev[1], &tmp_outputs[5]};
    void *UH_2[] = {&state_h_dev, &U_dev[2], &tmp_outputs[6]};
    void *UH_3[] = {&state_h_dev, &U_dev[3], &tmp_outputs[7]};
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)WI_0, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)WI_1, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)WI_2, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)WI_3, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)UH_0, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)UH_1, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)UH_2, COLUMNS_PER_BLOCK, stream_t);
    cudaLaunchKernel((const void *)gemv, dim3(hidden_size >> 5), dim3(256),
                     (void **)UH_3, COLUMNS_PER_BLOCK, stream_t);
    void *solve_args[] = {&tmp_outputs[0], &tmp_outputs[4],  &bias_dev[0],
                          &tmp_outputs[1], &tmp_outputs[5],  &bias_dev[1],
                          &tmp_outputs[2], &tmp_outputs[6],  &bias_dev[2],
                          &tmp_outputs[3], &tmp_outputs[7],  &bias_dev[3],
                          &state_c_dev,    &state_c_new_dev, &output_dev};
    cudaLaunchKernel((const void *)solve, dim3(1), dim3(hidden_size),
                     (void **)solve_args, 0, stream_t);
}
void LSTMCell::Close() {
    free(output_host);
    cudaFree(input_dev);
}
} // namespace mica::experiments::lstm