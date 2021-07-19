#include "LSTMNet.h"
#include <iostream>
namespace mica::experiments::lstm {
void LSTMNet::initLSTMNet(const std::vector<HostCellParams> &netParams) {
    for (int i = 0; i < num_layer; ++i) {
        cells[i].init(netParams[i]);
    }
}

void LSTMNet::compute(const std::vector<float *> input_devs) {
    for (int i = 0; i < num_step; ++i) {
        for (int j = 0; j < num_layer; ++j) {
            if (j != 0)
                cells[j].compute(cells[j - 1].state_h_dev);
            else
                cells[j].compute(input_devs[i]);
        }
    }
}

float *LSTMNet::getOutput() { return cells[num_layer - 1].getResult(); }

void LSTMNet::release() {
    for (auto cell : cells) {
        cell.Close();
    }
}

} // namespace mica::experiments::lstm