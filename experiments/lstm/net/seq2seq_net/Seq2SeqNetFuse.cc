#include "Seq2SeqNetFuse.h"
#include "Seq2SeqNetFuseFunctions.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mica::experiments::lstm {

void Seq2SeqNetFuse::release() {
    cudaFree(inputs_dev);
    cudaFree(input);
    cudaFree(model);
    cudaFree(output);
    free(output_host);
    cudaStreamDestroy(stream);
}
void Seq2SeqNetFuse::initSeq2SeqNet(
    const std::vector<LSTMNetHostParams> &netParams) {
    cudaMalloc(&inputs_dev,
               sizeof(float) * (enc_step + 1) * enc_layer * hidden_size +
                   sizeof(float) * enc_layer * hidden_size +
                   sizeof(float) * enc_step * hidden_size +
                   sizeof(float) * (dec_step + 1) * dec_layer * hidden_size +
                   sizeof(float) * dec_step * hidden_size +
                   sizeof(float) * dec_layer * hidden_size);

    cudaMalloc(&input, sizeof(WaveInputParams) *
                           (enc_step * enc_layer + dec_step * dec_layer));
    cudaMalloc(&model, sizeof(WaveModelParams) * (enc_layer + dec_layer));
    cudaMalloc(&output, sizeof(WaveOutputParams) *
                            (enc_layer * enc_step + dec_layer * dec_step));
    cudaStreamCreate(&stream);
    output_host = (float *)malloc(sizeof(float) * hidden_size);

    WaveInputParams *inputHost = (WaveInputParams *)malloc(
        sizeof(WaveInputParams) *
        (enc_step * enc_layer + dec_step * dec_layer));
    WaveModelParams *modelHost = (WaveModelParams *)malloc(
        sizeof(WaveModelParams) * (enc_layer + dec_layer));
    WaveOutputParams *outputHost = (WaveOutputParams *)malloc(
        sizeof(WaveOutputParams) *
        (enc_layer * enc_step + dec_layer * dec_step));

    state_c_s = inputs_dev + (dec_step + enc_step) * hidden_size;
    state_h_s = state_c_s + (dec_layer + enc_layer) * hidden_size;
    for (int i = 0; i < enc_step; ++i) {
        for (int j = 0; j < enc_layer; ++j) {
            if (j == 0) {
                (inputHost + i * enc_layer + j)->input_i =
                    inputs_dev + i * hidden_size; //?
            } else {
                (inputHost + i * enc_layer + j)->input_i =
                    state_h_s + ((i + 1) * enc_layer + j - 1) * hidden_size;
            }
            (inputHost + i * enc_layer + j)->input_h =
                state_h_s + (i * enc_layer + j) * hidden_size;
            (outputHost + i * enc_layer + j)->state_c =
                state_c_s + j * hidden_size;
            (outputHost + i * enc_layer + j)->state_h =
                state_h_s + ((i + 1) * enc_layer + j) * hidden_size;
        }
    }

    for (int i = 0; i < dec_step; ++i) {
        for (int j = 0; j < dec_layer; ++j) {
            if (j == 0 && i == 0) {
                (inputHost + i * dec_layer + j + enc_layer * enc_step)
                    ->input_i =
                    (outputHost + enc_layer * enc_step - 1)->state_c;
            } else {
                (inputHost + i * dec_layer + j + enc_layer * enc_step)
                    ->input_i =
                    (outputHost + i * dec_layer + j + enc_layer * enc_step - 1)
                        ->state_h;
            }
            (inputHost + i * dec_layer + j + enc_layer * enc_step)->input_h =
                state_h_s +
                (i * dec_layer + j + enc_layer * enc_step) * hidden_size;
            (outputHost + i * dec_layer + j + enc_layer * enc_step)->state_c =
                state_c_s + (j + enc_layer) * hidden_size;
            (outputHost + i * dec_layer + j + enc_layer * enc_step)->state_h =
                state_h_s +
                ((i + 1) * dec_layer + j + (enc_step + 1) * enc_layer) *
                    hidden_size;
        }
    }

    // copydata
    for (int i = 0; i < enc_step; ++i) {
        cudaMemcpy(inputs_dev + i * hidden_size,
                   netParams[0].inputs + i * hidden_size,
                   sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < (enc_layer + dec_layer); ++i) {
        cudaMemcpy(state_c_s + i * hidden_size, netParams[i].state_c_s,
                   sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i <= enc_step; ++i) {
        for (int j = 0; j < enc_layer; ++j) {
            cudaMemcpy(state_h_s + (i * enc_layer + j) * hidden_size,
                       netParams[j].state_h_s, sizeof(float) * hidden_size,
                       cudaMemcpyHostToDevice);
        }
    }

    for (int i = 0; i <= dec_step; ++i) {
        for (int j = 0; j < dec_layer; ++j) {
            cudaMemcpy(state_h_s + (i * dec_layer + j + enc_layer * enc_step) *
                                       hidden_size,
                       netParams[j + enc_layer].state_h_s,
                       sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
        }
    }

    for (int i = 0; i < (dec_layer + enc_layer); ++i) {
        memcpy((modelHost + i)->weight_w, netParams[i].weights_w,
               sizeof(float) * hidden_size * hidden_size * 4);
        memcpy((modelHost + i)->weight_u, netParams[i].weights_u,
               sizeof(float) * hidden_size * hidden_size * 4);
        memcpy((modelHost + i)->bias, netParams[i].bias_s,
               sizeof(float) * hidden_size * 4);
    }

    for (int i = 0; i < (dec_layer + enc_layer); ++i) {
        float *hostTempWS =
            (float *)malloc(sizeof(float4) * hidden_size * hidden_size);
        float *hostTempUS =
            (float *)malloc(sizeof(float4) * hidden_size * hidden_size);
        float *hostTempBiasS = (float *)malloc(sizeof(float4) * hidden_size);
        for (int m = 0; m < hidden_size; m++)
            for (int n = 0; n < hidden_size; n++) {
                for (int k = 0; k < 4; k++) {

                    hostTempWS[k * 128 * 128 + n * 128 + m] =
                        netParams[i].weights_w[(n * hidden_size + m) * 4 + k];
                    hostTempUS[k * 128 * 128 + n * 128 + m] =
                        netParams[i].weights_u[(n * hidden_size + m) * 4 + k];
                    hostTempBiasS[k * 128 + m] = netParams[i].bias_s[m * 4 + k];
                }
            }
        memcpy((modelHost + i)->weight_ws, hostTempWS,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((modelHost + i)->weight_us, hostTempUS,
               sizeof(float4) * hidden_size * hidden_size);
        memcpy((modelHost + i)->biass, hostTempBiasS,
               sizeof(float4) * hidden_size);

        free(hostTempWS);
        free(hostTempUS);
        free(hostTempBiasS);
    }

    cudaMemcpy(input, inputHost,
               sizeof(WaveInputParams) *
                   (enc_step * enc_layer + dec_step * dec_layer),
               cudaMemcpyHostToDevice);
    cudaMemcpy(model, modelHost,
               sizeof(WaveModelParams) * (enc_layer + dec_layer),
               cudaMemcpyHostToDevice);
    cudaMemcpy(output, outputHost,
               sizeof(WaveOutputParams) *
                   (enc_step * enc_layer + dec_step * dec_layer),
               cudaMemcpyHostToDevice);
    free(inputHost);

    free(modelHost);
    free(outputHost);
}

// void Seq2SeqNetFuse::compute() {
//     void *arg_s[] = {&input, &model, &output};
//     cudaLaunchKernel((void *)seq2seq_enc_wave0, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_enc_wave1, dim3(8), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave2, dim3(12), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave3, dim3(16), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave4, dim3(20), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave5, dim3(24), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave6, dim3(28), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave7, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave8, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave9, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave10, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave11, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave12, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave13, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave14, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave15, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave16, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave17, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave18, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave19, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave20, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave21, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave22, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave23, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave24, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave25, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave26, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave27, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave28, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave29, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave30, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave31, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave32, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave33, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave34, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave35, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave36, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave37, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave38, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave39, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave40, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave41, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave42, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave43, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave44, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave45, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave46, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave47, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave48, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave49, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave50, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave51, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave52, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave53, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave54, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave55, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave56, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave57, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave58, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave59, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave60, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave61, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave62, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave63, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave64, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave65, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave66, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave67, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave68, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave69, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave70, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave71, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave72, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave73, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave74, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave75, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave76, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave77, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave78, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave79, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave80, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave81, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave82, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave83, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave84, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave85, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave86, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave87, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave88, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave89, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave90, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave91, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave92, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave93, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave94, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave95, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave96, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave97, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave98, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave99, dim3(32), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave100, dim3(28), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave101, dim3(24), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave102, dim3(20), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave103, dim3(16), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave104, dim3(12), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave105, dim3(8), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_enc_wave106, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);

//     cudaLaunchKernel((void *)seq2seq_dec_wave0, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave1, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave2, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave3, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave4, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave5, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave6, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave7, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave8, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave9, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave10, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave11, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave12, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave13, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave14, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave15, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave16, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave17, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave18, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave19, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave20, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave21, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave22, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave23, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave24, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave25, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave26, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave27, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave28, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave29, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave30, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave31, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave32, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave33, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave34, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave35, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave36, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave37, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave38, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave39, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave40, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave41, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave42, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave43, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave44, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave45, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave46, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave47, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave48, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave49, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave50, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave51, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave52, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave53, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave54, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave55, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave56, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave57, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave58, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave59, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave60, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave61, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave62, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave63, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave64, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave65, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave66, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave67, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave68, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave69, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave70, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave71, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave72, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave73, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave74, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave75, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave76, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave77, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave78, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave79, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave80, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave81, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave82, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave83, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave84, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave85, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave86, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave87, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave88, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave89, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave90, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave91, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave92, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave93, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave94, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave95, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave96, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave97, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave98, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave99, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave100, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave101, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave102, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave103, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave104, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave105, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave106, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave107, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave108, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave109, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave110, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave111, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave112, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave113, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave114, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave115, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave116, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave117, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave118, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaLaunchKernel((void *)seq2seq_dec_wave119, dim3(4), dim3(256),
//                      (void **)arg_s, 0, stream);
//     cudaDeviceSynchronize();
// }

void Seq2SeqNetFuse::computeAndSolve() {
    void *arg_s[] = {&input, &model, &output};

    // auto t = cudaLaunchKernel((void *)wave_compute_0, dim3(4), dim3(256),
    //                           (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_solve_0, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_1, dim3(8), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_solve_1, dim3(8), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    //    cudaLaunchKernel((void *)wave_compute_2, dim3(12), dim3(256),
    //                      (void **)arg_s, 0, stream);
    //     cudaLaunchKernel((void *)wave_solve_2, dim3(12), dim3(256), (void
    //     **)arg_s,
    //                      0, stream);

    auto t = cudaLaunchKernel((void *)wave_compute_3, dim3(16), dim3(256),
                              (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_3, dim3(16), dim3(256), (void **)arg_s,
                     0, stream);

    // cudaLaunchKernel((void *)wave_compute_4, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_4, dim3(20), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    //  auto  t = cudaLaunchKernel((void *)wave_compute_5, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_5, dim3(24), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    t = cudaDeviceSynchronize();
    std::cout << t << std::endl;
    // auto  t = cudaLaunchKernel((void *)wave_compute_6, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);
    //     std::cout << t << std::endl;

    // t = cudaDeviceSynchronize();
    // std::cout << t << std::endl;
    // cudaLaunchKernel((void *)wave_solve_6, dim3(28), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_7, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_7, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_8, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_8, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_9, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_9, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_10, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_10, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_11, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_11, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_12, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_12, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_13, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_13, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_14, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_14, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_15, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_15, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_16, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_16, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_17, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_17, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_18, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_18, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_19, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_19, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_20, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_20, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_21, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_21, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_22, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_22, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_23, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_23, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_24, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_24, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_25, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_25, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_26, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_26, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_27, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_27, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_28, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_28, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_29, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_29, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_30, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_30, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_31, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_31, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_32, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_32, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_33, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_33, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_34, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_34, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_35, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_35, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_36, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_36, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_37, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_37, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_38, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_38, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_39, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_39, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_40, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_40, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_41, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_41, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_42, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_42, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_43, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_43, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_44, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_44, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_45, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_45, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_46, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_46, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_47, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_47, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_48, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_48, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_49, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_49, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_50, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_50, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_51, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_51, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_52, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_52, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_53, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_53, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_54, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_54, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_55, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_55, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_56, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_56, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_57, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_57, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_58, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_58, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_59, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_59, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_60, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_60, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_61, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_61, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_62, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_62, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_63, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_63, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_64, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_64, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_65, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_65, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_66, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_66, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_67, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_67, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_68, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_68, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_69, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_69, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_70, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_70, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_71, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_71, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_72, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_72, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_73, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_73, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_74, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_74, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_75, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_75, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_76, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_76, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_77, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_77, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_78, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_78, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_79, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_79, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_80, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_80, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_81, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_81, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_82, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_82, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_83, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_83, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_84, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_84, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_85, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_85, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_86, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_86, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_87, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_87, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_88, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_88, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_89, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_89, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_90, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_90, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_91, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_91, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_92, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_92, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_93, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_93, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_94, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_94, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_95, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_95, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_96, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_96, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_97, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_97, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_98, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_98, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_99, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_99, dim3(32), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_100, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_100, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_compute_101, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_101, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_compute_102, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_102, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_compute_103, dim3(16), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_103, dim3(16), dim3(256),
    //                  (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_compute_104, dim3(12), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_104, dim3(12), dim3(256),
    //                  (void **)arg_s, 0, stream);

    // cudaLaunchKernel((void *)wave_compute_105, dim3(8), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_105, dim3(8), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_106, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_106, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)wave_compute_107, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_107, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_108, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_108, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_109, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_109, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_110, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_110, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_111, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_111, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_112, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_112, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_113, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_113, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_114, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_114, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_115, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_115, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_116, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_116, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_117, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_117, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_118, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_118, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_119, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_119, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_120, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_120, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_121, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_121, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_122, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_122, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_123, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_123, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_124, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_124, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_125, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_125, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_126, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_126, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_127, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_127, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_128, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_128, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_129, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_129, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_130, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_130, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_131, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_131, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_132, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_132, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_133, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_133, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_134, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_134, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_135, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_135, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_136, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_136, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_137, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_137, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_138, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_138, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_139, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_139, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_140, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_140, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_141, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_141, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_142, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_142, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_143, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_143, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_144, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_144, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_145, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_145, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_146, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_146, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_147, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_147, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_148, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_148, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_149, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_149, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_150, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_150, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_151, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_151, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_152, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_152, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_153, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_153, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_154, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_154, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_155, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_155, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_156, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_156, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_157, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_157, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_158, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_158, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_159, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_159, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_160, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_160, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_161, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_161, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_162, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_162, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_163, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_163, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_164, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_164, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_165, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_165, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_166, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_166, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_167, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_167, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_168, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_168, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_169, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_169, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_170, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_170, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_171, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_171, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_172, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_172, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_173, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_173, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_174, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_174, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_175, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_175, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_176, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_176, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_177, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_177, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_178, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_178, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_179, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_179, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_180, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_180, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_181, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_181, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_182, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_182, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_183, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_183, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_184, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_184, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_185, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_185, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_186, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_186, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_187, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_187, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_188, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_188, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_189, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_189, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_190, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_190, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_191, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_191, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_192, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_192, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_193, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_193, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_194, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_194, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_195, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_195, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_196, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_196, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_197, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_197, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_198, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_198, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_199, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_199, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_200, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_200, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_201, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_201, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_202, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_202, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_203, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_203, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_204, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_204, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_205, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_205, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_206, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_206, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_207, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_207, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_208, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_208, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_209, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_209, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_210, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_210, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_211, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_211, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_212, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_212, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_213, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_213, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_214, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_214, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_215, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_215, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_216, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_216, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_217, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_217, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_218, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_218, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_219, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_219, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_220, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_220, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_221, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_221, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_222, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_222, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_223, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_223, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_224, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_224, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_225, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_225, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);
    // cudaLaunchKernel((void *)wave_compute_226, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)wave_solve_226, dim3(4), dim3(256), (void
    // **)arg_s,
    //                  0, stream);

    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_0, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_0, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_1, dim3(8), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_1, dim3(8), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_2, dim3(12), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_2, dim3(12), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_3, dim3(16), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_3, dim3(16), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_4, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_4, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_5, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_5, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_6, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_6, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_7, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_7, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_8, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_8, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_9, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_9, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_10, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_10, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_11, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_11, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_12, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_12, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_13, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_13, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_14, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_14, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_15, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_15, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_16, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_16, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_17, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_17, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_18, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_18, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_19, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_19, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_20, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_20, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_21, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_21, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_22, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_22, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_23, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_23, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_24, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_24, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_25, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_25, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_26, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_26, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_27, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_27, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_28, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_28, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_29, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_29, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_30, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_30, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_31, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_31, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_32, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_32, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_33, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_33, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_34, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_34, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_35, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_35, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_36, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_36, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_37, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_37, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_38, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_38, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_39, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_39, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_40, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_40, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_41, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_41, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_42, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_42, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_43, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_43, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_44, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_44, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_45, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_45, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_46, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_46, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_47, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_47, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_48, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_48, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_49, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_49, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_50, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_50, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_51, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_51, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_52, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_52, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_53, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_53, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_54, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_54, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_55, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_55, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_56, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_56, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_57, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_57, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_58, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_58, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_59, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_59, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_60, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_60, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_61, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_61, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_62, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_62, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_63, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_63, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_64, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_64, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_65, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_65, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_66, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_66, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_67, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_67, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_68, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_68, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_69, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_69, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_70, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_70, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_71, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_71, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_72, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_72, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_73, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_73, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_74, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_74, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_75, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_75, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_76, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_76, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_77, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_77, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_78, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_78, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_79, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_79, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_80, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_80, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_81, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_81, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_82, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_82, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_83, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_83, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_84, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_84, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_85, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_85, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_86, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_86, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_87, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_87, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_88, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_88, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_89, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_89, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_90, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_90, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_91, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_91, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_92, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_92, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_93, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_93, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_94, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_94, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_95, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_95, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_96, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_96, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_97, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_97, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_98, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_98, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_99, dim3(32),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_99, dim3(32), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_100, dim3(28),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_100, dim3(28), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_101, dim3(24),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_101, dim3(24), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_102, dim3(20),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_102, dim3(20), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_103, dim3(16),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_103, dim3(16), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_104, dim3(12),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_104, dim3(12), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_105, dim3(8),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_105, dim3(8), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_compute_106, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_enc_wave_solve_106, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_0, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_0, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_1, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_1, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_2, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_2, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_3, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_3, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_4, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_4, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_5, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_5, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_6, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_6, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_7, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_7, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_8, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_8, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_9, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_9, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_10, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_10, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_11, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_11, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_12, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_12, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_13, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_13, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_14, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_14, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_15, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_15, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_16, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_16, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_17, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_17, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_18, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_18, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_19, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_19, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_20, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_20, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_21, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_21, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_22, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_22, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_23, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_23, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_24, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_24, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_25, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_25, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_26, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_26, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_27, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_27, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_28, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_28, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_29, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_29, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_30, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_30, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_31, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_31, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_32, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_32, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_33, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_33, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_34, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_34, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_35, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_35, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_36, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_36, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_37, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_37, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_38, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_38, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_39, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_39, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_40, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_40, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_41, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_41, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_42, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_42, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_43, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_43, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_44, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_44, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_45, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_45, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_46, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_46, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_47, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_47, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_48, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_48, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_49, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_49, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_50, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_50, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_51, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_51, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_52, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_52, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_53, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_53, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_54, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_54, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_55, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_55, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_56, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_56, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_57, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_57, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_58, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_58, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_59, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_59, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_60, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_60, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_61, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_61, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_62, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_62, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_63, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_63, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_64, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_64, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_65, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_65, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_66, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_66, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_67, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_67, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_68, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_68, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_69, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_69, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_70, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_70, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_71, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_71, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_72, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_72, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_73, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_73, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_74, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_74, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_75, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_75, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_76, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_76, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_77, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_77, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_78, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_78, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_79, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_79, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_80, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_80, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_81, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_81, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_82, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_82, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_83, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_83, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_84, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_84, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_85, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_85, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_86, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_86, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_87, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_87, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_88, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_88, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_89, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_89, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_90, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_90, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_91, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_91, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_92, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_92, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_93, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_93, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_94, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_94, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_95, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_95, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_96, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_96, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_97, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_97, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_98, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_98, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_99, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_99, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_100, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_100, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_101, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_101, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_102, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_102, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_103, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_103, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_104, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_104, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_105, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_105, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_106, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_106, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_107, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_107, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_108, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_108, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_109, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_109, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_110, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_110, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_111, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_111, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_112, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_112, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_113, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_113, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_114, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_114, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_115, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_115, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_116, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_116, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_117, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_117, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_118, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_118, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_compute_119, dim3(4),
    // dim3(256),
    //                  (void **)arg_s, 0, stream);
    // cudaLaunchKernel((void *)seq2seq_dec_wave_solve_119, dim3(4), dim3(256),
    //                  (void **)arg_s, 0, stream);
    cudaDeviceSynchronize();
}

} // namespace mica::experiments::lstm