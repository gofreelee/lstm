#include "../../net/lstm_cell_net/LSTMNet.h"
#include "../LstmTest.h"
#include <asm-generic/errno-base.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

#include "cuda_runtime.h"

#include <memory>
#include <unistd.h>
#include <utility>

namespace mica::experiments::lstm {

namespace unittest {

using namespace std;

const size_t num_layer = 8, num_step = 8;
const size_t batch_size = 1, hidden_size = 256;

TEST_F(LstmTest, LSTMNet_Test) {
    float *all_one_input =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    for (int i = 0; i < batch_size * hidden_size; ++i)
        all_one_input[i] = 1.0000f;
    const float *input[] = {all_one_input, all_one_input, all_one_input,
                            all_one_input, all_one_input, all_one_input,
                            all_one_input, all_one_input};

    float *all_zero_state =
        (float *)malloc(sizeof(float) * batch_size * hidden_size * 2);
    for (int i = 0; i < batch_size * hidden_size * 2; ++i)
        all_zero_state[i] = 0.000f;
    const float *init_state[] = {all_zero_state, all_zero_state, all_zero_state,
                                 all_zero_state, all_zero_state, all_zero_state,
                                 all_zero_state, all_zero_state};

    const float *W[] = {W_0_0, W_0_1, W_0_2, W_0_3, W_1_0, W_1_1, W_1_2, W_1_3,
                        W_2_0, W_2_1, W_2_2, W_2_3, W_3_0, W_3_1, W_3_2, W_3_3,
                        W_4_0, W_4_1, W_4_2, W_4_3, W_5_0, W_5_1, W_5_2, W_5_3,
                        W_6_0, W_6_1, W_6_2, W_6_3, W_7_0, W_7_1, W_7_2, W_7_3};

    const float *U[] = {U_0_0, U_0_1, U_0_2, U_0_3, U_1_0, U_1_1, U_1_2, U_1_3,
                        U_2_0, U_2_1, U_2_2, U_2_3, U_3_0, U_3_1, U_3_2, U_3_3,
                        U_4_0, U_4_1, U_4_2, U_4_3, U_5_0, U_5_1, U_5_2, U_5_3,
                        U_6_0, U_6_1, U_6_2, U_6_3, U_7_0, U_7_1, U_7_2, U_7_3};

    const float *bias[] = {
        bias_0_0, bias_0_1, bias_0_2, bias_0_3, bias_1_0, bias_1_1, bias_1_2,
        bias_1_3, bias_2_0, bias_2_1, bias_2_2, bias_2_3, bias_3_0, bias_3_1,
        bias_3_2, bias_3_3, bias_4_0, bias_4_1, bias_4_2, bias_4_3, bias_5_0,
        bias_5_1, bias_5_2, bias_5_3, bias_6_0, bias_6_1, bias_6_2, bias_6_3,
        bias_7_0, bias_7_1, bias_7_2, bias_7_3};
    std::vector<HostCellParams> cellParams;
    float *_W[8], *_U[8], *_bias[8];
    for (int i = 0; i < 8; ++i) {
        _W[i] = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
        _U[i] = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
        _bias[i] = (float *)malloc(sizeof(float) * 4 * hidden_size);
    }

    for (int i = 0; i < 8; ++i) {
        memcpy(_W[i], W[i * 4], sizeof(float) * hidden_size * hidden_size);
        memcpy(_W[i] + hidden_size * hidden_size, W[i * 4 + 1],
               sizeof(float) * hidden_size * hidden_size);
        memcpy(_W[i] + 2 * hidden_size * hidden_size, W[i * 4 + 2],
               sizeof(float) * hidden_size * hidden_size);
        memcpy(_W[i] + 3 * hidden_size * hidden_size, W[i * 4 + 3],
               sizeof(float) * hidden_size * hidden_size);

        memcpy(_U[i], U[i * 4], sizeof(float) * hidden_size * hidden_size);
        memcpy(_U[i] + hidden_size * hidden_size, U[i * 4 + 1],
               sizeof(float) * hidden_size * hidden_size);
        memcpy(_U[i] + 2 * hidden_size * hidden_size, U[i * 4 + 2],
               sizeof(float) * hidden_size * hidden_size);
        memcpy(_U[i] + 3 * hidden_size * hidden_size, U[i * 4 + 3],
               sizeof(float) * hidden_size * hidden_size);

        memcpy(_bias[i], bias[i * 4], sizeof(float) * hidden_size);
        memcpy(_bias[i] + hidden_size, bias[i * 4 + 1],
               sizeof(float) * hidden_size);
        memcpy(_bias[i] + 2 * hidden_size, bias[i * 4 + 2],
               sizeof(float) * hidden_size);
        memcpy(_bias[i] + 3 * hidden_size, bias[i * 4 + 3],
               sizeof(float) * hidden_size);

        HostCellParams param = {all_zero_state, all_zero_state, _W[i], _U[i],
                                _bias[i]};
        cellParams.push_back(param);
    }
    float *input_dev;
    cudaMalloc(&input_dev, sizeof(float) * hidden_size);
    cudaMemcpy(input_dev, all_one_input, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    std::vector<float *> input_devs(num_layer, input_dev);
    shared_ptr<LSTMNet> impl(
        new LSTMNet(num_step, num_layer, hidden_size, hidden_size));
    impl->initLSTMNet(cellParams);
    impl->compute(input_devs);
    cudaDeviceSynchronize();
    ASSERT_OUTPUT(impl->getOutput());
    impl->release();
    free(all_one_input);
    free(all_zero_state);
    cudaFree(input_dev);
    for (int i = 0; i < 4; ++i) {
        free(_W[i]);
        free(_U[i]);
        free(_bias[i]);
    }
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
