#include "LSTM100_10.h"
#include "WavefrontFunctionArgs.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mica::experiments::lstm {
void LSTM100_10::init(const std::vector<LSTMNetHostParams> &params) {
    cudaMalloc(&inputs_dev,
               sizeof(float) * num_layer *
                       (8 * hidden_size * hidden_size + 5 * hidden_size) +
                   sizeof(float) * num_step * hidden_size +
                   sizeof(float) * hidden_size * num_layer * (num_step + 1));
    cudaMalloc(&wi, sizeof(float4) * num_step * num_layer * hidden_size);
    cudaMalloc(&uh, sizeof(float4) * num_step * num_layer * hidden_size);
    cudaMalloc(&waveInputParams_dev,
               sizeof(WaveInputParams) * num_step * num_layer);
    cudaMalloc(&waveModelParams_dev, sizeof(WaveModelParams) * num_layer);
    cudaMalloc(&waveOutputParams_dev,
               sizeof(WaveOutputParams) * num_step * num_layer);

    WaveInputParams *waveInputParams = (WaveInputParams *)malloc(
        sizeof(WaveInputParams) * num_step * num_layer);
    WaveModelParams *waveModelParams =
        (WaveModelParams *)malloc(sizeof(WaveModelParams) * num_layer);
    WaveOutputParams *waveOutputParams = (WaveOutputParams *)malloc(
        sizeof(WaveOutputParams) * num_step * num_layer);
    cudaStreamCreate(&stream);
    output_host = (float *)malloc(sizeof(float) * hidden_size);
    state_c_s = inputs_dev + num_step * hidden_size;
    state_h_s = state_c_s + num_layer * hidden_size;
    weights_w = reinterpret_cast<float4 *>(
        state_h_s + (num_step + 1) * num_layer * hidden_size);
    weights_u = weights_w + num_layer * hidden_size * hidden_size;
    bias_s = weights_u + num_layer * hidden_size * hidden_size;
    for (int i = 0; i < num_step; ++i) {
        cudaMemcpy(inputs_dev + i * hidden_size, params[0].inputs,
                   sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    }
    // cudaMemcpy(state_h_s + i * hidden_size, params[i].state_h_s,
    //           sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    for (int i = 0; i < num_layer; ++i) {
        cudaMemcpy(state_c_s + i * hidden_size, params[i].state_c_s,
                   sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
        cudaMemcpy(
            weights_w + i * hidden_size * hidden_size, params[i].weights_w,
            sizeof(float4) * hidden_size * hidden_size, cudaMemcpyHostToDevice);
        cudaMemcpy(
            weights_u + i * hidden_size * hidden_size, params[i].weights_u,
            sizeof(float4) * hidden_size * hidden_size, cudaMemcpyHostToDevice);
        cudaMemcpy(bias_s + i * hidden_size, params[i].bias_s,
                   sizeof(float4) * hidden_size, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < num_layer; ++i) {
        for (int j = 0; j <= num_step; ++j) {
            cudaMemcpy(state_h_s + hidden_size * (j * num_layer + i),
                       params[i].state_h_s, sizeof(float) * hidden_size,
                       cudaMemcpyHostToDevice);
        }
    }
    for (int i = 0; i < num_step; ++i) {
        for (int j = 0; j < num_layer; ++j) {
            if (j == 0)
                (waveInputParams + i * num_layer + j)->input_i =
                    inputs_dev + i * hidden_size;
            else
                (waveInputParams + i * num_layer + j)->input_i =
                    state_h_s + ((i + 1) * num_layer + j - 1) * hidden_size;

            (waveInputParams + i * num_layer + j)->input_h =
                state_h_s + (i * num_layer + j) * hidden_size;
            (waveOutputParams + i * num_layer + j)->wi =
                wi + (i * num_layer + j) * hidden_size;
            (waveOutputParams + i * num_layer + j)->uh =
                uh + (i * num_layer + j) * hidden_size;
            (waveOutputParams + i * num_layer + j)->state_c =
                state_c_s + j * hidden_size;
            (waveOutputParams + i * num_layer + j)->state_h =
                state_h_s + ((i + 1) * num_layer + j) * hidden_size;
        }
    }

    for (int i = 0; i < num_layer; ++i) {

        memcpy((waveModelParams + i)->weight_w, params[i].weights_w,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParams + i)->weight_u, params[i].weights_u,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParams + i)->bias, params[i].bias_s,
               sizeof(float4) * hidden_size);
    }

    cudaMemcpy(waveInputParams_dev, waveInputParams,
               sizeof(WaveInputParams) * num_step * num_layer,
               cudaMemcpyHostToDevice);
    cudaMemcpy(waveModelParams_dev, waveModelParams,
               sizeof(WaveModelParams) * num_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(waveOutputParams_dev, waveOutputParams,
               sizeof(WaveOutputParams) * num_step * num_layer,
               cudaMemcpyHostToDevice);
    free(waveInputParams);
    free(waveModelParams);
    free(waveOutputParams);
}

void LSTM100_10::finalize() {
    cudaFree(inputs_dev);
    cudaFree(wi);
    cudaFree(uh);
    cudaFree(waveInputParams_dev);
    cudaFree(waveModelParams_dev);
    cudaFree(waveOutputParams_dev);
    free(output_host);
}
void LSTM100_10::compute() {
    void *arg_s[] = {&waveInputParams_dev, &waveModelParams_dev,
                     &waveOutputParams_dev};
    cudaLaunchKernel((void *)wave0, dim3(8), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave1, dim3(16), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave2, dim3(24), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave3, dim3(32), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave4, dim3(40), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave5, dim3(48), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave6, dim3(56), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave7, dim3(64), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave8, dim3(72), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave9, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave10, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave11, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave12, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave13, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave14, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave15, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave16, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave17, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave18, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave19, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave20, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave21, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave22, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave23, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave24, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave25, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave26, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave27, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave28, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave29, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave30, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave31, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave32, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave33, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave34, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave35, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave36, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave37, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave38, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave39, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave40, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave41, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave42, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave43, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave44, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave45, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave46, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave47, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave48, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave49, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave50, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave51, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave52, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave53, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave54, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave55, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave56, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave57, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave58, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave59, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave60, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave61, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave62, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave63, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave64, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave65, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave66, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave67, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave68, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave69, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave70, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave71, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave72, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave73, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave74, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave75, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave76, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave77, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave78, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave79, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave80, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave81, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave82, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave83, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave84, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave85, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave86, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave87, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave88, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave89, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave90, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave91, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave92, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave93, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave94, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave95, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave96, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave97, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave98, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave99, dim3(80), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave100, dim3(72), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave101, dim3(64), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave102, dim3(56), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave103, dim3(48), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave104, dim3(40), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave105, dim3(32), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave106, dim3(24), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave107, dim3(16), dim3(256), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave108, dim3(8), dim3(256), (void **)arg_s, 0,
                     stream);
    cudaDeviceSynchronize();
}
} // namespace mica::experiments::lstm