#pragma once

#include <absl/status/status.h>
#include <memory>
#include <string_view>

namespace mica::experiments::lstm {

//
// The interfaces of the inference solver for LSTM.
class LSTMInference {
  public:
    // TODO: specify the types of both the input and the output format.
    virtual ~LSTMInference() = default;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Close() = 0;
    virtual absl::Status Solve(const std::string_view input) = 0;
    virtual absl::Status Fetch() = 0;
};

std::unique_ptr<LSTMInference> NewNaiveLSTMInference();

} // namespace mica::experiments::lstm