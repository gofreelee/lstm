#pragma once
#include "cuda_runtime.h"
#include "LSTM100_10Experiment.h"
#include <cstddef>
#include <vector>

namespace mica::experiments::lstm {
class LSTM100_10_4W_4U_16blocksExperiment: public LSTM100_10Experiment {
  public:
    virtual void computeAndSolve();
};

} // namespace mica::experiments::lstm