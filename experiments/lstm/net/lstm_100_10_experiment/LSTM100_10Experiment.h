#pragma once
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10/LSTM100_10.h"
#include "net/lstm_cell_net_100_10/WavefrontParams.h"
#include <cstddef>
#include <vector>

namespace mica::experiments::lstm {
class LSTM100_10Experiment {
  protected:
    /* data */
    size_t num_step, num_layer, hidden_size, batch_size;
    float *inputs_dev;
    float *state_c_s, *state_h_s;
    float4 *weights_w, *weights_u, *bias_s;
    float4 *wi, *uh;
    WaveInputParams *waveInputParams_dev;
    WaveModelParams *waveModelParams_dev;
    WaveOutputParams *waveOutputParams_dev;
    cudaStream_t stream;
    float *output_host;

  public:
    LSTM100_10Experiment()
        : num_step(lstm_100_10_num_step), num_layer(lstm_100_10_num_layer),
          hidden_size(lstm_100_10_hidden_size),
          batch_size(lstm_100_10_batch_size) {}
    LSTM100_10Experiment(size_t num_step_, size_t num_layer_)
        : num_step(num_step_), num_layer(num_layer_),
          hidden_size(lstm_100_10_hidden_size),
          batch_size(lstm_100_10_batch_size) {}
    void init(const std::vector<LSTMNetHostParams> &parmas);
    virtual void computeAndSolve();

    void finalize();
    float *getOutput() {
        cudaMemcpy(output_host,
                   state_h_s + hidden_size * ((num_step + 1) * num_layer - 1),
                   sizeof(float) * hidden_size, cudaMemcpyDeviceToHost);
        return output_host;
    }
};

} // namespace mica::experiments::lstm