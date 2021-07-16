#pragma once

// #include <absl/status/status.h>
#include <memory>
#include <string_view>

namespace mica::experiments::lstm {

struct HostParam {
    const float **input;
    const float **init_state;
    const float **W;
    const float **U;
    const float **bias;
};

//
// The interfaces of the inference solver for LSTM.
class LSTMInference {

  protected:
    LSTMInference(size_t num_step, size_t num_layer, size_t batch_size,
                  size_t hidden_size)
        : num_step(num_step), num_layer(num_layer), batch_size(batch_size),
          hidden_size(hidden_size) {}
    size_t num_step, num_layer;
    size_t batch_size, hidden_size;

  public:
    // TODO: specify the types of both the input and the output format.
    virtual ~LSTMInference() = default;
    virtual void Initialize(const HostParam &param) = 0;
    virtual void Close() = 0;
    virtual void Solve(const HostParam &input) = 0;
    virtual const float *Fetch() = 0;
};

std::unique_ptr<LSTMInference> NewNaiveLSTMInference();

} // namespace mica::experiments::lstm