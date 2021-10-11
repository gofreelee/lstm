#include "../LstmTest.h"
#include "net/seq2seq_net/Seq2SeqArgs.h"
#include "net/seq2seq_net/Seq2SeqNetFuse.h"
#include <asm-generic/errno-base.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include "sys/time.h"

#include "cuda_runtime.h"

#include <bitset>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <utility>

using namespace std;

const size_t enc_layer = 8, dec_layer = 4;
const size_t enc_step = 100, dec_step = 30;
const size_t batch_size = 1, input_size = 128, hidden_size = 128;

void ReadAll(const std::string &file, float *result, size_t size) {
    std::ifstream in(file, ios::in | ios::binary);
    in.read((char *)result, size);
    in.close();
}

int main(int argc, char *argv[]) {
    using namespace mica::experiments::lstm;
    enum {
        kLoop = 1000,
    };
    float *inputs =
        (float *)malloc(sizeof(float) * batch_size * enc_step * input_size);
    float *h_state =
        (float *)malloc(sizeof(float) * (enc_layer + dec_layer) * hidden_size);
    float *c_state =
        (float *)malloc(sizeof(float) * (enc_layer + dec_layer) * hidden_size);
    float *W = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                               hidden_size * hidden_size);
    float *U = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                               input_size * hidden_size);
    float *bias = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                                  hidden_size);
    float *_W = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                                hidden_size * hidden_size);
    float *_U = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                                input_size * hidden_size);
    float *_bias = (float *)malloc(sizeof(float) * 4 * (enc_layer + dec_layer) *
                                   hidden_size);
    float *output = (float *)malloc(sizeof(float) * hidden_size);
    ReadAll("./data/input.data", inputs,
            sizeof(float) * batch_size * enc_step * input_size);
    ReadAll("./data/w.data", W,
            sizeof(float) * 4 * (enc_layer + dec_layer) * hidden_size *
                hidden_size);
    ReadAll("./data/u.data", U,
            sizeof(float) * 4 * (enc_layer + dec_layer) * input_size *
                hidden_size);
    ReadAll("./data/bias.data", bias,
            sizeof(float) * 4 * (enc_layer + dec_layer) * hidden_size);
    ReadAll("./data/output.data", output, sizeof(float) * hidden_size);
    memset(h_state, 0, sizeof(float) * (enc_layer + dec_layer) * hidden_size);
    memset(c_state, 0, sizeof(float) * (enc_layer + dec_layer) * hidden_size);
    for (int i = 0; i < (enc_layer + dec_layer); ++i) {
        float *once_W_fuse = _W + i * 4 * hidden_size * hidden_size;
        float *once_W = W + i * 4 * hidden_size * hidden_size;
        float *once_U_fuse = _U + i * 4 * hidden_size * hidden_size;
        float *once_U = U + i * 4 * hidden_size * hidden_size;
        for (int j = 0; j < hidden_size * hidden_size; ++j) {
            once_W_fuse[j * 4] = once_W[j];
            once_W_fuse[j * 4 + 1] = once_W[j + hidden_size * hidden_size];
            once_W_fuse[j * 4 + 2] = once_W[j + hidden_size * hidden_size * 2];
            once_W_fuse[j * 4 + 3] = once_W[j + hidden_size * hidden_size * 3];
            once_U_fuse[j * 4] = once_U[j];
            once_U_fuse[j * 4 + 1] = once_U[j + hidden_size * hidden_size];
            once_U_fuse[j * 4 + 2] = once_U[j + hidden_size * hidden_size * 2];
            once_U_fuse[j * 4 + 3] = once_U[j + hidden_size * hidden_size * 3];
        }
    }
    for (int i = 0; i < (enc_layer + dec_layer); ++i) {
        float *once_bias_fuse = _bias + i * 4 * hidden_size;
        float *once_bias = bias + i * 4 * hidden_size;
        for (int j = 0; j < hidden_size; ++j) {
            once_bias_fuse[j * 4] = once_bias[j];
            once_bias_fuse[j * 4 + 1] = once_bias[j + hidden_size];
            once_bias_fuse[j * 4 + 2] = once_bias[j + hidden_size * 2];
            once_bias_fuse[j * 4 + 3] = once_bias[j + hidden_size * 3];
        }
    }
    std::vector<LSTMNetHostParams> cellParams;
    for (int i = 0; i < enc_layer + dec_layer; ++i) {
        float *h_state_layer = h_state + hidden_size * i;
        float *c_state_layer = c_state + hidden_size * i;
        float *W_layer = _W + 4 * hidden_size * hidden_size * i;
        float *U_layer = _U + 4 * input_size * hidden_size * i;
        float *bias_layer = _bias + 4 * hidden_size * i;
        LSTMNetHostParams param = {inputs,  c_state_layer, h_state_layer,
                                   W_layer, U_layer,       bias_layer};
        cellParams.push_back(param);
    }

    shared_ptr<Seq2SeqNetFuse> impl(new Seq2SeqNetFuse(
        enc_step, dec_step, enc_layer, dec_layer, input_size, hidden_size));
    impl->initSeq2SeqNet(cellParams);
    float *inputs_dev;
    cudaMalloc(&inputs_dev, sizeof(float) * batch_size * enc_step * input_size);
    cudaMemcpy(inputs_dev, inputs,
               sizeof(float) * batch_size * enc_step * input_size,
               cudaMemcpyHostToDevice);
    impl->computeAndSolve();
    float *result = impl->getOutput();
    for (unsigned i = 0; i < hidden_size; i++) {
        float diff = fabs(result[i] - output[i]);
        diff = diff > 0.0001f ? diff : 0.0f;
        if (diff != 0.0f)
            return 0;
    }
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

    for (int i = 0; i < 5; i++) {
        // Different, input/output memcpy time
        impl->computeAndSolve();
    }
    timeval time_start;
    timeval time_end;
    long long walltimes = 0;
    std::vector<double> recordVec;
    double maxTime = 0;
    double minTime = 10;
    for (int i = 0; i < 100; ++i) {
        gettimeofday(&time_start, NULL);
        impl->computeAndSolve();
        gettimeofday(&time_end, NULL);
        long long once_time = (time_end.tv_sec - time_start.tv_sec) * 1000000 +
                              time_end.tv_usec - time_start.tv_usec;
    }
    for (int i = 0; i < kLoop; i++) {
        // Different, input/output memcpy time
        gettimeofday(&time_start, NULL);
        impl->computeAndSolve();
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
    impl->release();
    free(inputs);
    free(h_state);
    free(c_state);
    free(W);
    free(U);
    free(bias);
    free(output);
    cudaFree(inputs_dev);
}
