#include "LSTMCellF4.h"
#define COLUMNS_PER_BLOCK 32

void gem4v(const float *__restrict__ input, const float4 *__restrict__ weight,
           float4 *__restrict__ output);
void solve_gem4v_res(float4 *__restrict__ wi, float4 *__restrict__ uh,
                     float4 *bias, float *state_c, float *state_h);
namespace mica::experiments::lstm {
void LSTMCellF4::init(const HostCellParams &params) {
    cudaMalloc(&state_c_dev, sizeof(float) * (4 * hidden_size * input_size +
                                              4 * hidden_size * hidden_size +
                                              6 * hidden_size));
    output_host = (float *)malloc(sizeof(float) * hidden_size);
    state_h_dev = state_c_dev + hidden_size;
    W_dev = reinterpret_cast<float4 *>(state_h_dev + hidden_size);
    U_dev = W_dev + hidden_size * input_size;
    bias_dev = U_dev + hidden_size * input_size;
    cudaMemcpy(state_c_dev, params.init_state_c, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_h_dev, params.init_state_h, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(W_dev, params.W, sizeof(float) * hidden_size * input_size * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(U_dev, params.U, sizeof(float) * hidden_size * input_size * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_dev, params.bias, sizeof(float) * hidden_size * 4,
               cudaMemcpyHostToDevice);
    cudaStreamCreate(&stream_t);
}
void LSTMCellF4::compute(float *input_dev) {
    float4 *tmp_outputs[2];
    for (int i = 0; i < 2; ++i)
        cudaMalloc(&tmp_outputs[i], sizeof(float4) * hidden_size);
    void *WI[] = {&input_dev, &W_dev, &tmp_outputs[0]};
    void *UH[] = {&state_h_dev, &U_dev, &tmp_outputs[1]};
    void *solve_args[] = {&tmp_outputs[0], &tmp_outputs[1], &bias_dev,
                          &state_c_dev, &state_h_dev};
    cudaLaunchKernel((const void *)gem4v, dim3(hidden_size >> 5),
                     dim3(hidden_size), (void **)WI,
                     COLUMNS_PER_BLOCK * sizeof(float4), stream_t);
    cudaLaunchKernel((const void *)gem4v, dim3(hidden_size >> 5),
                     dim3(hidden_size), (void **)UH,
                     COLUMNS_PER_BLOCK * sizeof(float4), stream_t);
    cudaLaunchKernel((const void *)solve_gem4v_res, dim3(1), dim3(hidden_size),
                     (void **)solve_args, 0, stream_t);
    cudaDeviceSynchronize();
    cudaFree(tmp_outputs[0]);
    cudaFree(tmp_outputs[1]);
}

void LSTMCellF4::Close() {
    free(output_host);
    cudaFree(state_c_dev);
    cudaStreamDestroy(stream_t);
}

float *LSTMCellF4::getResult() {
    cudaMemcpy(output_host, state_h_dev, sizeof(float) * hidden_size,
               cudaMemcpyDeviceToHost);
    return output_host;
}

} // namespace mica::experiments::lstm