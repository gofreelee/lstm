#include "LSTM100_10_BS16.h"
#include "WavefrontFunctionArgsBS16.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mica::experiments::lstm {
void LSTM100_10_BS16::init(const std::vector<LSTMNetHostParams> &params) {
    cudaMalloc(
        &inputs_dev,
        sizeof(float4) * num_layer *
                (8 * hidden_size * hidden_size + 4 * hidden_size) +
            sizeof(float4) * num_step * hidden_size * 4 +
            sizeof(float4) * hidden_size * num_layer * (num_step + 1) * 4 +
            sizeof(float4) * num_layer * hidden_size * 4 +
            16 * sizeof(float) * num_layer * hidden_size +
            16 * sizeof(float) * num_step * hidden_size +
            16 * sizeof(float) * num_layer * (num_step + 1) * hidden_size);
    cudaMalloc(&wi, sizeof(float4) * num_step * num_layer * hidden_size);
    cudaMalloc(&uh, sizeof(float4) * num_step * num_layer * hidden_size);
    cudaMalloc(&WaveInputParamsBS16_dev,
               sizeof(WaveInputParamsBS16) * num_step * num_layer);
    cudaMalloc(&WaveModelParamsBS16_dev,
               sizeof(WaveModelParamsBS16) * num_layer);
    cudaMalloc(&WaveOutputParamsBS16_dev,
               sizeof(WaveOutputParamsBS16) * num_step * num_layer);

    WaveInputParamsBS16 *waveInputParamsBS16 = (WaveInputParamsBS16 *)malloc(
        sizeof(WaveInputParamsBS16) * num_step * num_layer);
    WaveModelParamsBS16 *waveModelParamsBS16 =
        (WaveModelParamsBS16 *)malloc(sizeof(WaveModelParamsBS16) * num_layer);
    WaveOutputParamsBS16 *waveOutputParamsBS16 = (WaveOutputParamsBS16 *)malloc(
        sizeof(WaveOutputParamsBS16) * num_step * num_layer);
    cudaStreamCreate(&stream);
    output_host = (float4 *)malloc(sizeof(float4) * hidden_size * 4);
    state_c_s = inputs_dev + num_step * hidden_size * 4;
    state_h_s = state_c_s + num_layer * hidden_size * 4;
    weights_w = reinterpret_cast<float4 *>(
        state_h_s + (num_step + 1) * num_layer * hidden_size * 4);
    weights_u = weights_w + num_layer * hidden_size * hidden_size;
    bias_s = weights_u + num_layer * hidden_size * hidden_size;
    inputs_dev_s =
        reinterpret_cast<float *>(bias_s) + num_layer * hidden_size * 4;
    state_c_dev_s = inputs_dev_s + num_step * hidden_size * 16;
    state_h_dev_s = state_c_dev_s + num_layer * hidden_size * 16;
    for (int i = 0; i < num_step; ++i) {
        cudaMemcpy(inputs_dev + i * hidden_size * 4, params[0].inputs,
                   sizeof(float4) * hidden_size * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(inputs_dev_s + i * hidden_size * 16, params[0].inputs,
                    sizeof(float) * hidden_size * 16, cudaMemcpyHostToDevice);
    }
   
    for (int i = 0; i < num_layer; ++i) {
        cudaMemcpy(state_c_s + i * hidden_size * 4, params[i].state_c_s,
                   sizeof(float4) * hidden_size * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(state_c_dev_s + i * hidden_size * 16, params[i].state_c_s,
                   sizeof(float) * hidden_size * 16, cudaMemcpyHostToDevice);
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
            cudaMemcpy(state_h_s + hidden_size * (j * num_layer + i) * 4,
                       params[i].state_h_s, sizeof(float4) * hidden_size * 4,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(state_h_dev_s + hidden_size * (j * num_layer + i) * 16,
                       params[i].state_h_s, sizeof(float) * hidden_size * 16,
                       cudaMemcpyHostToDevice);
        }
    }
    for (int i = 0; i < num_step; ++i) {
        for (int j = 0; j < num_layer; ++j) {
            if (j == 0) {
                (waveInputParamsBS16 + i * num_layer + j)->input_i =
                    inputs_dev + i * hidden_size * 4;
                (waveInputParamsBS16 + i * num_layer + j)->input_i_f1 =
                    inputs_dev_s + i * hidden_size * 16;
            } else {
                (waveInputParamsBS16 + i * num_layer + j)->input_i =
                    state_h_s + ((i + 1) * num_layer + j - 1) * hidden_size * 4;
                (waveInputParamsBS16 + i * num_layer + j)->input_i_f1 = 
                    state_h_dev_s + ((i + 1) * num_layer + j - 1) * hidden_size * 16;
            }
            (waveInputParamsBS16 + i * num_layer + j)->input_h =
                state_h_s + (i * num_layer + j) * hidden_size * 4;
            (waveInputParamsBS16 + i * num_layer + j)->input_h_f1 = 
                state_h_dev_s + (i * num_layer + j) * hidden_size * 16;
            (waveOutputParamsBS16 + i * num_layer + j)->wi =
                wi + (i * num_layer + j) * hidden_size;
            (waveOutputParamsBS16 + i * num_layer + j)->uh =
                uh + (i * num_layer + j) * hidden_size;
            (waveOutputParamsBS16 + i * num_layer + j)->state_c =
                state_c_s + j * hidden_size * 4;
            (waveOutputParamsBS16 + i * num_layer + j)->state_c_f1 =
                state_c_dev_s + j * hidden_size * 16;
            (waveOutputParamsBS16 + i * num_layer + j)->state_h =
                state_h_s + ((i + 1) * num_layer + j) * hidden_size * 4;
            (waveOutputParamsBS16 + i * num_layer + j)->state_h_f1 =
                state_h_dev_s + ((i + 1) * num_layer + j) * hidden_size * 16;
        }
    }

    for (int i = 0; i < num_layer; ++i) {

        memcpy((waveModelParamsBS16 + i)->weight_w, params[i].weights_w,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParamsBS16 + i)->weight_u, params[i].weights_u,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParamsBS16 + i)->bias, params[i].bias_s,
               sizeof(float4) * hidden_size);
    }


 for (int i = 0; i < num_layer; ++i) {

        float *hostTempWS =
            (float *)malloc(sizeof(float4) * hidden_size * hidden_size);
        float *hostTempUS =
            (float *)malloc(sizeof(float4) * hidden_size * hidden_size);
        float *hostTempBiasS = (float *)malloc(sizeof(float4) * hidden_size);
        for (int m = 0; m < hidden_size; m++)
            for (int n = 0; n < hidden_size; n++) {
                for (int k = 0; k < 4; k++) {

                    hostTempWS[k * 256 * 256 + n * 256 + m] =
                        params[i].weights_w[(n * hidden_size + m) * 4 + k];
                    hostTempUS[k * 256 * 256 + n * 256 + m] =
                        params[i].weights_u[(n * hidden_size + m) * 4 + k];
                    hostTempBiasS[k * 256 + m] = params[i].bias_s[m * 4 + k];
                }
            }

        memcpy((waveModelParamsBS16 + i)->weight_ws, hostTempWS,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParamsBS16 + i)->weight_us, hostTempUS,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((waveModelParamsBS16 + i)->biass, hostTempBiasS,
               sizeof(float4) * hidden_size);

        free(hostTempWS);
        free(hostTempUS);
        free(hostTempBiasS);
    }



    cudaMemcpy(WaveInputParamsBS16_dev, waveInputParamsBS16,
               sizeof(WaveInputParamsBS16) * num_step * num_layer,
               cudaMemcpyHostToDevice);
    cudaMemcpy(WaveModelParamsBS16_dev, waveModelParamsBS16,
               sizeof(WaveModelParamsBS16) * num_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(WaveOutputParamsBS16_dev, waveOutputParamsBS16,
               sizeof(WaveOutputParamsBS16) * num_step * num_layer,
               cudaMemcpyHostToDevice);
    // for(int i = 0;i < hidden_size * hidden_size; ++i)
    // {
    //     std::cout << waveModelParamsBS4->weight_w[i].x << " "
    //               << waveModelParamsBS4->weight_w[i].y << " "
    //               << waveModelParamsBS4->weight_w[i].z << " "
    //               << waveModelParamsBS4->weight_w[i].w << std::endl;
    // }
    free(waveInputParamsBS16);
    free(waveModelParamsBS16);
    free(waveOutputParamsBS16);
}

void LSTM100_10_BS16::finalize() {
    cudaFree(inputs_dev);
    cudaFree(wi);
    cudaFree(uh);
    cudaFree(WaveInputParamsBS16_dev);
    cudaFree(WaveModelParamsBS16_dev);
    cudaFree(WaveOutputParamsBS16_dev);
    free(output_host);
}
void LSTM100_10_BS16::compute() {
    void *arg_s[] = {&WaveInputParamsBS16_dev, &WaveModelParamsBS16_dev,
                     &WaveOutputParamsBS16_dev};
    cudaLaunchKernel((void *)wave0, dim3(8), dim3(256, 4), (void **)arg_s, 0,
                     stream);
    cudaLaunchKernel((void *)wave1, dim3(16), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave2, dim3(24), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave3, dim3(32), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave4, dim3(40), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave5, dim3(48), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave6, dim3(56), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave7, dim3(64), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave8, dim3(72), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave9, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave10, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave11, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave12, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave13, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave14, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave15, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave16, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave17, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave18, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave19, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave20, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave21, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave22, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave23, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave24, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave25, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave26, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave27, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave28, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave29, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave30, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave31, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave32, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave33, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave34, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave35, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave36, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave37, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave38, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave39, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave40, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave41, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave42, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave43, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave44, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave45, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave46, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave47, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave48, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave49, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave50, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave51, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave52, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave53, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave54, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave55, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave56, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave57, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave58, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave59, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave60, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave61, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave62, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave63, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave64, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave65, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave66, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave67, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave68, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave69, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave70, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave71, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave72, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave73, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave74, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave75, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave76, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave77, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave78, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave79, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave80, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave81, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave82, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave83, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave84, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave85, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave86, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave87, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave88, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave89, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave90, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave91, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave92, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave93, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave94, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave95, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave96, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave97, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave98, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave99, dim3(80), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave100, dim3(72), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave101, dim3(64), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave102, dim3(56), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave103, dim3(48), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave104, dim3(40), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave105, dim3(32), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave106, dim3(24), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave107, dim3(16), dim3(256, 4), (void **)arg_s, 0,
                     stream);

    cudaLaunchKernel((void *)wave108, dim3(8), dim3(256, 4), (void **)arg_s, 0,
                     stream);
    cudaDeviceSynchronize();
}
} // namespace mica::experiments::lstm