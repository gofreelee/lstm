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

#include "cuda_runtime.h"

#include <bitset>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <utility>
namespace mica::experiments::lstm {

namespace unittest {

using namespace std;

const size_t enc_layer = 8, dec_layer = 4;
const size_t enc_step = 100, dec_step = 30;
const size_t batch_size = 1, input_size = 128, hidden_size = 128;

void ReadAll(const std::string &file, float *result, size_t size) {
    std::ifstream in(file, ios::in | ios::binary);
    in.read((char *)result, size);
    in.close();
}

TEST_F(LstmTest, Seq2SeqNetTest) {
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
    // Different, input/output memcpy time
    impl->compute();
    float *result = impl->getOutput();
    for (unsigned i = 0; i < hidden_size; i++) {
        float diff = fabs(result[i] - output[i]);
        diff = diff > 0.0001f ? diff : 0.0f;
        ASSERT_FLOAT_EQ(diff, 0.0f);
    }
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

} // namespace unittest
} // namespace mica::experiments::lstm

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
