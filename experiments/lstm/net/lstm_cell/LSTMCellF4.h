#pragma once
#include "../Lstm.h"

#include "cuda_runtime.h"

namespace mica::experiments::lstm {
struct HostCellParams {
    const float *init_state_c;
    const float *init_state_h;
    const float *W;
    const float *U;
    const float *bias;
};
class LSTMCellF4 {

  public:
    LSTMCellF4(size_t inputSize, size_t hiddenSize)
        : input_size(inputSize), hidden_size(hiddenSize) {}
    void init(const HostCellParams &param);
    void compute(float *input_dev);
    float *getResult();
    void Close();

  private:
    size_t input_size, hidden_size;
    float4 *W_dev, *U_dev, *bias_dev;
    float *state_h_dev;
    float *output_host;
    float *state_c_dev;
    cudaStream_t stream_t;
};
} // namespace mica::experiments::lstm
