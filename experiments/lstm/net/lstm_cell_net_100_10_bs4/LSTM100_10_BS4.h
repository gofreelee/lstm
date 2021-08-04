#pragma once
#include "WavefrontParamsBS4.h"
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10/LSTM100_10.h"
#include <cstddef>
#include <vector>

namespace mica::experiments::lstm {
class LSTM100_10_BS4 {
    enum {
        lstm_100_10_num_step = 100,
        lstm_100_10_num_layer = 10,
        lstm_100_10_hidden_size = 256,
        lstm_100_10_batch_size = 4,
    };

  private:
    /* data */
    size_t num_step, num_layer, hidden_size, batch_size;
    float4 *inputs_dev;
    float4 *state_c_s, *state_h_s;
    float4 *weights_w, *weights_u, *bias_s;
    float4 *wi, *uh;
    WaveInputParamsBS4 *WaveInputParamsBS4_dev;
    WaveModelParamsBS4 *WaveModelParamsBS4_dev;
    WaveOutputParamsBS4 *WaveOutputParamsBS4_dev;
    cudaStream_t stream;
    float4 *output_host;

  public:
    LSTM100_10_BS4()
        : num_step(lstm_100_10_num_step), num_layer(lstm_100_10_num_layer),
          hidden_size(lstm_100_10_hidden_size),
          batch_size(lstm_100_10_batch_size) {}
    void init(const std::vector<LSTMNetHostParams> &parmas);
    void compute();
    void finalize();
    float4 *getOutput() {
        cudaMemcpy(output_host,
                   state_h_s + hidden_size * ((num_step + 1) * num_layer - 1),
                   sizeof(float4) * hidden_size, cudaMemcpyDeviceToHost);
        return output_host;
    }
};

} // namespace mica::experiments::lstm