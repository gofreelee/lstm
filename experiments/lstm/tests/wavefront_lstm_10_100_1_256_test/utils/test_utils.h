#pragma once
#include "../../../net/wavefront_lstm_10_100_1_256/include/RammerLike2.h"
#include "../../LstmTest.h"
#include "cuda_runtime.h"
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace mica::experiments::lstm {

std::vector<const float **> readInputParams(int fd, int hidden_size,
                                            int batch_size, int num_layer);
void freeParams(const std::vector<const float **> &params);
float *readExpectedResult(int fd, int hidden_size);
} // namespace mica::experiments::lstm