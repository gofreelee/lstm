#pragma once
#include "../../LstmTest.h"
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10/LSTM100_10.h"
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace mica::experiments::lstm {

std::vector<const float **> readInputParams(int fd, int hidden_size,
                                            int batch_size, int num_layer);
std::vector<LSTMNetHostParams> readInputParamsFuse(int fd, int hidden_size,
                                                   int batch_size,
                                                   int num_layer, int num_step);
void freeParams(const std::vector<const float **> &params);
void freeParamsFuse(const std::vector<LSTMNetHostParams> &parmas);
float *readExpectedResult(int fd, int hidden_size);
} // namespace mica::experiments::lstm