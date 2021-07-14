#include "model/alexnet.h"
#include "op/utils.h"
#include "utils/test_utils.h"
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

using namespace mica::experiments::models;

class AlexnetTest : public ::testing::Test {
  protected:
    virtual void SetUp() override {
        output_.resize(sizeof(AlexnetOutput) / sizeof(float));
        input_ = RandomData(sizeof(*device_input_) / sizeof(float));
        auto state = RandomData(sizeof(*state_) / sizeof(float));
        expect_ = ReadAll("alexnet.out", "./data");

        ASSERT(expect_.size() == output_.size());
        CUDA_CHECK(cudaMalloc(&device_input_, sizeof(*device_input_)));
        CUDA_CHECK(cudaMalloc(&state_, sizeof(*state_)));
        CUDA_CHECK(cudaMemcpy(state_, state.data(), sizeof(*state_),
                              cudaMemcpyHostToDevice));
    }

    virtual void TearDown() override {
        cudaFree(state_);
        cudaFree(device_input_);
    }

    std::vector<float> input_;
    std::vector<float> output_;
    std::vector<float> expect_;
    AlexnetState *state_;
    AlexnetInput *device_input_;
};

TEST_F(AlexnetTest, TestSingleInfer) {
    enum {
        kLoop = 100,
    };

    for (int i = 0; i < kLoop; ++i) {
        CUDA_CHECK(cudaMemcpy(device_input_, input_.data(),
                              sizeof(float) * input_.size(),
                              cudaMemcpyHostToDevice));
        const AlexnetOutput *output = AlexnetInfer(device_input_, state_);
        CUDA_CHECK(cudaMemcpy(output_.data(), output, sizeof(*output),
                              cudaMemcpyDeviceToHost));
    }

    for (unsigned i = 0; i < output_.size(); ++i) {
        ASSERT_LT(fabs(expect_[i] - output_[i]), 0.0001);
    }

    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda, 0);

    for (int i = 0; i < kLoop; ++i) {
        CUDA_CHECK(cudaMemcpy(device_input_, input_.data(),
                              sizeof(float) * input_.size(),
                              cudaMemcpyHostToDevice));
        const AlexnetOutput *output = AlexnetInfer(device_input_, state_);
        CUDA_CHECK(cudaMemcpy(output_.data(), output, sizeof(*output),
                              cudaMemcpyDeviceToHost));
    }

    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);
    float cuda_event_timepassed = 0;
    cudaEventElapsedTime(&cuda_event_timepassed, start_cuda, stop_cuda);
    printf("Elapsed time (ms): %f\n", cuda_event_timepassed / kLoop);
}
