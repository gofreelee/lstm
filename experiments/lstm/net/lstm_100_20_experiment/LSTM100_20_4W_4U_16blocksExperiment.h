#pragma once
#include "cuda_runtime.h"
#include "LSTM100_20Experiment.h"
#include <cstddef>
#include <vector>

namespace mica::experiments::lstm {
class LSTM100_20_4W_4U_16blocksExperiment: public LSTM100_20Experiment {
  public:
    virtual void computeAndSolve();
};

} // namespace mica::experiments::lstm