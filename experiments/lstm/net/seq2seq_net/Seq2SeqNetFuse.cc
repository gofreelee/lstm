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

void Seq2SeqNetFuse::compute() {
    void *arg_s[] = {&input, &model, &output};
    cudaLaunchKernel((void *)seq2seq_enc_wave0, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_enc_wave1, dim3(8), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave2, dim3(12), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave3, dim3(16), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave4, dim3(20), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave5, dim3(24), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave6, dim3(28), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave7, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave8, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave9, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave10, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave11, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave12, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave13, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave14, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave15, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave16, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave17, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave18, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave19, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave20, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave21, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave22, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave23, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave24, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave25, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave26, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave27, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave28, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave29, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave30, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave31, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave32, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave33, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave34, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave35, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave36, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave37, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave38, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave39, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave40, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave41, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave42, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave43, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave44, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave45, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave46, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave47, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave48, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave49, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave50, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave51, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave52, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave53, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave54, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave55, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave56, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave57, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave58, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave59, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave60, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave61, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave62, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave63, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave64, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave65, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave66, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave67, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave68, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave69, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave70, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave71, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave72, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave73, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave74, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave75, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave76, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave77, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave78, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave79, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave80, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave81, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave82, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave83, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave84, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave85, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave86, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave87, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave88, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave89, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave90, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave91, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave92, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave93, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave94, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave95, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave96, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave97, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave98, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave99, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave100, dim3(28), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave101, dim3(24), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave102, dim3(20), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave103, dim3(16), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave104, dim3(12), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave105, dim3(8), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_enc_wave106, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);

    cudaLaunchKernel((void *)seq2seq_dec_wave0, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave1, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave2, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave3, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave4, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave5, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave6, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave7, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave8, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave9, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave10, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave11, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave12, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave13, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave14, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave15, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave16, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave17, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave18, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave19, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave20, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave21, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave22, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave23, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave24, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave25, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave26, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave27, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave28, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave29, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave30, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave31, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave32, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave33, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave34, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave35, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave36, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave37, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave38, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave39, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave40, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave41, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave42, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave43, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave44, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave45, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave46, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave47, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave48, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave49, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave50, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave51, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave52, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave53, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave54, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave55, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave56, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave57, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave58, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave59, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave60, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave61, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave62, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave63, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave64, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave65, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave66, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave67, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave68, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave69, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave70, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave71, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave72, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave73, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave74, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave75, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave76, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave77, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave78, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave79, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave80, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave81, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave82, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave83, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave84, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave85, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave86, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave87, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave88, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave89, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave90, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave91, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave92, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave93, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave94, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave95, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave96, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave97, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave98, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave99, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave100, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave101, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave102, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave103, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave104, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave105, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave106, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave107, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave108, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave109, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave110, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave111, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave112, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave113, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave114, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave115, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave116, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave117, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave118, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)seq2seq_dec_wave119, dim3(4), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaDeviceSynchronize();
}
} // namespace mica::experiments::lstm