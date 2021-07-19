#include "lstm_cell_test.h"
#include "../../net/lstm_cell/LSTMCell.h"
#include <asm-generic/errno-base.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

#include "cuda_runtime.h"

#include <fstream>
#include <memory>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mica::experiments::lstm {

namespace unittest {

using namespace std;
const size_t batch_size = 1, hidden_size = 256;

void ReadAll(const std::string &file, const std::string &dir, float *result) {
    std::ifstream is(dir + '/' + file, std::ifstream::binary);
    char buf[4096];
    int offset = 0;
    while (!is.bad() && !is.eof()) {
        is.read(buf, sizeof(buf));
        size_t count = is.gcount();
        if (!count) {
            break;
        }
        memcpy(result + offset, buf, count);
        offset += count / sizeof(float);
    }
}

TEST_F(LstmTest, test_cell) {
    float *all_one_input =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    ReadAll("lstm.input", "./test_data/", all_one_input);

    float *all_zero_state =
        (float *)malloc(sizeof(float) * batch_size * hidden_size * 2);
    for (int i = 0; i < batch_size * hidden_size * 2; ++i)
        all_zero_state[i] = 0.000f;

    float *W1 = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    float *W = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    ReadAll("weighti", "./test_data/", W1);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            for (int k = 0; k < hidden_size; ++k) {
                W[i * hidden_size * hidden_size + j * hidden_size + k] =
                    W1[i * hidden_size * hidden_size + k * hidden_size + j];
            }
        }
    }
    float *U1 = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    float *U = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    ReadAll("weighth", "./test_data/", U1);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            for (int k = 0; k < hidden_size; ++k) {
                U[i * hidden_size * hidden_size + j * hidden_size + k] =
                    U1[i * hidden_size * hidden_size + k * hidden_size + j];
            }
        }
    }
    float *bias = (float *)malloc(sizeof(float) * 4 * hidden_size);
    ReadAll("bias", "./test_data/", bias);

    HostCellParams param = {all_zero_state, all_zero_state, W, U, bias};
    float *input_dev;
    cudaMalloc(&input_dev, sizeof(float) * hidden_size);
    cudaMemcpy(input_dev, all_one_input, sizeof(float) * hidden_size,
               cudaMemcpyHostToDevice);
    shared_ptr<LSTMCell> cell(new LSTMCell(hidden_size, hidden_size));
    cell->init(param);
    cell->compute(input_dev);
    cudaDeviceSynchronize();
    float *output = cell->getResult();
    float *expect_output = (float *)malloc(sizeof(float) * hidden_size);
    ReadAll("lstm.output", "./test_data/", expect_output);

    ASSERT_OUTPUT(cell->getResult(), expect_output);
    cell->Close();
    free(all_one_input);
    free(all_zero_state);
    free(W);
    free(U);
    free(W1);
    free(U1);
    free(bias);
    free(expect_output);
    cudaFree(input_dev);
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
