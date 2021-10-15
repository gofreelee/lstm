#include "cuda_runtime.h"
#include "net/lstm_100_20_experiment/LSTM100_20Experiment.h"
#include "net/lstm_100_20_experiment/LSTM100_20_1W_1U_32blocksExperiment.h"
#include "tests/wavefront_lstm_20_100_256_test/LstmTest_100_20.h"
#include "tests/wavefront_lstm_20_100_256_test/utils/test_utils.h"
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <utility>

#include <iostream>

namespace mica::experiments::lstm {

namespace unittest {

using namespace std;

const size_t num_layer = 10, num_step = 100;
const size_t batch_size = 1, hidden_size = 256;

TEST_F(LstmTest_100_20, FuseWaveFrontTest) {
    enum {
        kLoop = 1000,
    };
    int fd = open("./test_data/inputParams.bin", O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int result_fd = open("./test_data/expectResult.bin", O_CREAT | O_RDWR,
                         S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    std::vector<LSTMNetHostParams> params =
        readInputParamsFuse(fd, hidden_size, batch_size, num_layer, num_step);
    for (int i = 0; i < num_layer; ++i) {
        params.push_back(params[i]);
    }
    shared_ptr<LSTM100_20Experiment> impl(
        new LSTM100_20_1W_1U_32blocksExperiment());

    impl->init(params);
    cudaDeviceSynchronize();
    impl->computeAndSolve();
    cudaDeviceSynchronize();
    impl->finalize();
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
