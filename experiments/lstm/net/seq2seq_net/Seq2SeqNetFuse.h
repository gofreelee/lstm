#pragma once
#include "Seq2SeqArgs.h"
#include "cuda_runtime.h"
#include <vector>

namespace mica::experiments::lstm {

class Seq2SeqNetFuse {
  public:
    Seq2SeqNetFuse(size_t encStep, size_t decStep, size_t encLayer,
                   size_t decLayer, size_t inputSize, size_t hiddenSize)
        : enc_step(encStep), dec_step(decStep), enc_layer(encLayer),
          dec_layer(decLayer), input_size(inputSize), hidden_size(hiddenSize) {}
    void initSeq2SeqNet(const std::vector<LSTMNetHostParams> &netParams);
    void compute();
    void release();
    float *getOutput() {
        cudaMemcpy(output_host,
                   (state_h_s + ((enc_step + 1) * enc_layer +
                                 (dec_step + 1) * dec_layer - 1) *
                                    hidden_size),
                   sizeof(float) * hidden_size, cudaMemcpyDeviceToHost);
        return output_host;
    }

  private:
    size_t enc_step;
    size_t dec_step;
    size_t enc_layer;
    size_t dec_layer;
    size_t input_size;
    size_t hidden_size;

    float *inputs_dev;
    float *state_c_s, *state_h_s;
    float *output_host;
    cudaStream_t stream;

    WaveInputParams *input;
    WaveModelParams *model;
    WaveOutputParams *output;
};
} // namespace mica::experiments::lstm