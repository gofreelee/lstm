#include "LstmTest_100_10.h"
#include "cuda_runtime.h"
#include "net/lstm_cell_net_100_10_bs4/LSTM100_10_BS4.h"
#include "utils/test_utils.h"
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <sys/time.h>
#include <utility>

#include <iostream>

using namespace std;

const size_t num_layer = 10, num_step = 100;
const size_t batch_size = 4, hidden_size = 256;

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

    float *kExpected = readExpectedResult(result_fd, hidden_size);
    shared_ptr<LSTM100_10_BS4> impl(new LSTM100_10_BS4());
    impl->init(params);
    cudaDeviceSynchronize();
    impl->compute();
    cudaDeviceSynchronize();

    timeval time_start;
    timeval time_end;
    long long walltimes = 0;
    std::vector<double> recordVec;
    double maxTime = 0;
    double minTime = 10;
    for (int i = 0; i < 100; ++i) {
        impl->compute();
    }
    for (int i = 0; i < kLoop; i++) {
        // Different, input/output memcpy time
        gettimeofday(&time_start, NULL);
        impl->compute();
        gettimeofday(&time_end, NULL);
        long long once_time = (time_end.tv_sec - time_start.tv_sec) * 1000000 +
                              time_end.tv_usec - time_start.tv_usec;
        maxTime = (static_cast<double>(once_time) / 1000000) > maxTime
                      ? (static_cast<double>(once_time) / 1000000)
                      : maxTime;
        minTime = (static_cast<double>(once_time) / 1000000) < minTime
                      ? (static_cast<double>(once_time) / 1000000)
                      : minTime;
        walltimes += once_time;
    }
    double t = static_cast<double>(walltimes) / 1000000;
    std::cout << "Average Elapsed time (ms): " << t << std::endl;
    std::cout << "Max Elapsed time (ms) :" << maxTime * 1000 << "\n"
              << "Min Elapsed time (ms) :" << minTime * 1000 << std::endl;
    impl->finalize();
    freeParamsFuse(params);
}