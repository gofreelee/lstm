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

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
    impl->init(params[0], params[1], params[2], params[3], params[4]);
    for (int i = 0; i < 5; i++) {
        impl->compute(params[0], params[1], params[2], params[3], params[4]);
        cudaDeviceSynchronize();
    }
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda, 0);
    for (int i = 0; i < kLoop; i++) {
        // Different, input/output memcpy time
        impl->compute(params[0], params[1], params[2], params[3], params[4]);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    float cuda_event_timepassed = 0;
    cudaEventElapsedTime(&cuda_event_timepassed, start_cuda, stop_cuda);
    testing::internal::CaptureStdout();
    std::cout << "My test";
    std::string output = testing::internal::GetCapturedStdout();
    printf("Elapsed time (ms): %f\n", cuda_event_timepassed / kLoop);
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