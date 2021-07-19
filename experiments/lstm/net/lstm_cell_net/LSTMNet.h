#pragma once
#include "../lstm_cell/LSTMCell.h"
#include <vector>
namespace mica::experiments::lstm {
class LSTMNet {
  public:
    LSTMNet(size_t numStep, size_t numLayer, size_t inputSize,
            size_t hiddenSize)
        : num_step(numStep), num_layer(numLayer), input_size(inputSize),
          hidden_size(hiddenSize) {
        for (int i = 0; i < numLayer; ++i)
            cells.push_back(LSTMCell(inputSize, hiddenSize));
    }
    void initLSTMNet(const std::vector<HostCellParams> &netParams);
    void compute(const std::vector<float *> input_devs);
    void release();
    float *getOutput();

  private:
    size_t num_step, num_layer, input_size, hidden_size;
    std::vector<LSTMCell> cells;
};
} // namespace mica::experiments::lstm