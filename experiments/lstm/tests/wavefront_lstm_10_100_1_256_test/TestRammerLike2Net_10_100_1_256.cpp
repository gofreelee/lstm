#include "../../net/wavefront_lstm_10_100_1_256/include/RammerLike2.h"
#include "LstmTest_100_10.h"
#include "cuda_runtime.h"
#include "utils/test_utils.h"
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <utility>

#include <iostream>

namespace mica::experiments::lstm {

extern template class LstmRammerLike2Net<100, 10, 1, 256>;

namespace unittest {

using namespace std;

const size_t num_layer = 10, num_step = 100;
const size_t batch_size = 1, hidden_size = 256;

TEST_F(LstmTest_100_10, RammerLike2Net_100_10_1_256) {
    enum {
        kLoop = 1000,
    };
    int fd = open("./test_data/inputParams.bin", O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int result_fd = open("./test_data/expectResult.bin", O_CREAT | O_RDWR,
                         S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    std::vector<const float **> params =
        readInputParams(fd, hidden_size, batch_size, num_layer);
    float *kExpected = readExpectedResult(result_fd, hidden_size);
    shared_ptr<LstmCompute> impl(
        new LstmRammerLike2Net<num_step, num_layer, batch_size, hidden_size>);
    impl->init(params[0], params[1], params[2], params[3], params[4]);
    impl->compute(params[0], params[1], params[2], params[3], params[4]);
    cudaDeviceSynchronize();

    ASSERT_OUTPUT(impl->getOutput(), kExpected);
    impl->finalize();
    freeParams(params);
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}