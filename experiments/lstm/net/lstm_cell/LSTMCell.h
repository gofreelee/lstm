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
class LSTMCell {

  public:
    friend class LSTMNet;

    LSTMCell(size_t inputSize, size_t hiddenSize)
        : input_size(inputSize), hidden_size(hiddenSize) {}
    void init(const HostCellParams &param);
    void compute(float *input_dev);
    float *getResult() {
        cudaMemcpy(output_host, state_h_dev, sizeof(float) * hidden_size,
                   cudaMemcpyDeviceToHost);
        return output_host;
    }
    void Close();

  private:
    size_t input_size, hidden_size;
    float *W_dev[4], *U_dev[4], *bias_dev[4];
    float *state_h_dev;
    float *state_c_dev;
    float *output_host;

    cudaStream_t stream_t;
};
} // namespace mica::experiments::lstm
