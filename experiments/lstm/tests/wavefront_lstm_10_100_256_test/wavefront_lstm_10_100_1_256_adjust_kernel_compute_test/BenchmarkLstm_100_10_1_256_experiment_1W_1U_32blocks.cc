#include "cuda_runtime.h"
#include "tests/wavefront_lstm_10_100_256_test/LstmTest_100_10.h"
#include <cuda_profiler_api.h>

#include "net/lstm_100_10_experiment/LSTM100_10Experiment.h"
#include "net/lstm_100_10_experiment/LSTM100_10_1W_1U_32blocksExperiment.h"
#include "tests/wavefront_lstm_10_100_256_test/utils/test_utils.h"
#include <chrono>
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <sys/time.h>
#include <utility>

#include <iostream>

const size_t num_layer = 10, num_step = 100;
const size_t batch_size = 1, hidden_size = 256;

int main(int argc, char *argv[]) {

    using namespace mica::experiments::lstm;
    enum {
        kWarmup = 100,
        kLoop = 1000,
    };
    int fd = open("./test_data/inputParams.bin", O_CREAT | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int result_fd = open("./test_data/expectResult.bin", O_CREAT | O_RDWR,
                         S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    std::vector<LSTMNetHostParams> params =
        readInputParamsFuse(fd, hidden_size, batch_size, num_layer, num_step);

    float *kExpected = readExpectedResult(result_fd, hidden_size);
    std::shared_ptr<LSTM100_10Experiment> impl(
        new LSTM100_10_1W_1U_32blocksExperiment());
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
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;

    // time measurement
    impl->init(params);
    cudaDeviceSynchronize();

    for (int i = 0; i < kWarmup; ++i) {
        impl->computeAndSolve();
    }
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = std::numeric_limits<double>::min();
    double total_ms = 0.00000f;
    for (int i = 0; i < kLoop; i++) {
        // Different, input/output memcpy time
        auto start = std::chrono::steady_clock::now();
        impl->computeAndSolve();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double iteration_ms = elapsed.count();
        printf("Iteration time %f ms\n", iteration_ms);
        min_ms = std::min(iteration_ms, min_ms);
        max_ms = std::max(iteration_ms, max_ms);
        total_ms = total_ms + iteration_ms;
    }
    printf("Sumamry: [min, max, mean] = [%f, %f, %f] ms\n", min_ms, max_ms,
           total_ms / kLoop);
    impl->finalize();
}