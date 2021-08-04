#include "LstmTest_100_10.h"
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10_bs4/LSTM100_10_BS4.h"
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
const size_t batch_size = 4, hidden_size = 256;

TEST_F(LstmTest_100_10, FuseWaveFrontBS4Test) {
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
    shared_ptr<LSTM100_10_BS4> impl(new LSTM100_10_BS4());
    impl->init(params);
    cudaDeviceSynchronize();
    impl->compute();
    cudaDeviceSynchronize();
    float *result = (float *)malloc(sizeof(float) * batch_size * hidden_size);
    float4 *output = impl->getOutput();
    for (int i = 0; i < hidden_size; ++i) {
        float4 currV = output[i];
        result[i] = currV.x;
        result[i + hidden_size] = currV.y;
        result[i + hidden_size * 2] = currV.z;
        result[i + hidden_size * 3] = currV.w;
    }

    for (int i = 0; i < batch_size; ++i)
        ASSERT_OUTPUT(result + i * hidden_size, kExpected);
    impl->finalize();
    freeParamsFuse(params);
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}