#include "cuda_runtime.h"
#include "tests/wavefront_lstm_20_100_256_test/LstmTest_100_20.h"
#include <cuda_profiler_api.h>

#include "net/lstm_100_20_experiment/LSTM100_20Experiment.h"
#include "net/lstm_100_20_experiment/LSTM100_20_4W_4U_16blocksExperiment.h"
#include "tests/wavefront_lstm_20_100_256_test/utils/test_utils.h"
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
    float *kExpected = readExpectedResult(result_fd, hidden_size);
    std::shared_ptr<LSTM100_20Experiment> impl(
        new LSTM100_20_4W_4U_16blocksExperiment());
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
    cudaEvent_t start, stop, start_i, stop_i;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    // time measurement
    cudaEventRecord(start);
    impl->init(params);
    cudaDeviceSynchronize();

    timeval time_start;
    timeval time_end;
    long long walltimes = 0;
    std::vector<double> recordVec;
    double maxTime = 0;
    double minTime = 10;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start_i, 0);
        impl->computeAndSolve();
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        if (ms_i > ms_max)
            ms_max = ms_i;
        if (ms_i < ms_min)
            ms_min = ms_i;
    }
    for (int i = 0; i < kLoop; i++) {
        // Different, input/output memcpy time
        cudaEventRecord(start_i, 0);
        impl->computeAndSolve();
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        if (ms_i > ms_max)
            ms_max = ms_i;
        if (ms_i < ms_min)
            ms_min = ms_i;
    }
    cudaProfilerStop();
    // time measurement

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n", ms_min, ms_max,
           ms_total / kLoop);
    impl->finalize();
}