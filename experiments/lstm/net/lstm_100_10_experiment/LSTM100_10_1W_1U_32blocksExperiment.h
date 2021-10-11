#pragma once
#include "cuda_runtime.h"
#include "LSTM100_10Experiment.h"
#include <cstddef>
#include <vector>

namespace mica::experiments::lstm {
class LSTM100_10_1W_1U_32blocksExperiment: public LSTM100_10Experiment {
  public:
    virtual void computeAndSolve();
};

} // namespace mica::experiments::lstm