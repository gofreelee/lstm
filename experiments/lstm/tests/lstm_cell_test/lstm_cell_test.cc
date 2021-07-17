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

#include <memory>
#include <unistd.h>
#include <utility>

namespace mica::experiments::lstm {

namespace unittest {

using namespace std;
const size_t batch_size = 1, hidden_size = 256;

TEST_F(LstmTest, test_cell) {
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

    float *W = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    memcpy(W, W_0_0, sizeof(float) * hidden_size * hidden_size);
    memcpy(W + hidden_size * hidden_size, W_0_1,
           sizeof(float) * hidden_size * hidden_size);
    memcpy(W + 2 * hidden_size * hidden_size, W_0_2,
           sizeof(float) * hidden_size * hidden_size);
    memcpy(W + 3 * hidden_size * hidden_size, W_0_3,
           sizeof(float) * hidden_size * hidden_size);

    float *U = (float *)malloc(sizeof(float) * 4 * hidden_size * hidden_size);
    memcpy(U, U_0_0, sizeof(float) * hidden_size * hidden_size);
    memcpy(U + hidden_size * hidden_size, U_0_1,
           sizeof(float) * hidden_size * hidden_size);
    memcpy(U + 2 * hidden_size * hidden_size, U_0_2,
           sizeof(float) * hidden_size * hidden_size);
    memcpy(U + 3 * hidden_size * hidden_size, U_0_3,
           sizeof(float) * hidden_size * hidden_size);

    float *bias = (float *)malloc(sizeof(float) * 4 * hidden_size);
    memcpy(bias, bias_0_0, sizeof(float) * hidden_size);
    memcpy(bias + hidden_size, bias_0_1, sizeof(float) * hidden_size);
    memcpy(bias + 2 * hidden_size, bias_0_2, sizeof(float) * hidden_size);
    memcpy(bias + 3 * hidden_size, bias_0_3, sizeof(float) * hidden_size);

    HostCellParams param = {all_one_input, all_zero_state, all_zero_state, W, U,
                            bias};
    shared_ptr<LSTMCell> cell(new LSTMCell(hidden_size, hidden_size));
    cell->init(param);
    cell->compute();
    cudaDeviceSynchronize();
    ASSERT_OUTPUT(cell->getResult());
    cell->Close();
    free(all_one_input);
    free(all_zero_state);
}

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
