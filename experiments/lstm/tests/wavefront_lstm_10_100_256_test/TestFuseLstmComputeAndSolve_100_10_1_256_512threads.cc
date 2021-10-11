#include "LstmTest_100_10.h"
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10/LSTM100_10.h"
#include "utils/test_utils.h"
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

TEST_F(LstmTest_100_10, FuseWaveFrontTest) {
    enum {
        kLoop = 1000,
    };

    int fd = open("./test_data/inputParams.bin", O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int result_fd = open("./test_data/expectResult.bin", O_CREAT | O_RDWR,
                         S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    std::vector<LSTMNetHostParams> params =
        readInputParamsFuse(fd, hidden_size, batch_size, num_layer, num_step);

    float *kExpected = readExpectedResult(result_fd, hidden_size);
    shared_ptr<LSTM100_10> impl(new LSTM100_10());

    impl->init(params);

    cudaDeviceSynchronize();

    impl->computeAndSolve512();
    cudaDeviceSynchronize();
    ASSERT_OUTPUT(impl->getOutput(), kExpected);
    impl->finalize();
    freeParamsFuse(params);
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
