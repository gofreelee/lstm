#pragma once
#include "../Lstm.h"
#include <cstdlib>

namespace mica::experiments::lstm {

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
class LstmOneKernelNet : public LSTMInference {
  public:
    LstmOneKernelNet(size_t x = 0, size_t y = 0, size_t z = 0, size_t k = 0)
        : LSTMInference(t_num_step, t_num_layer, t_batch_size, t_hidden_size) {}

    bool hasFetchOutput{false};

    void Initialize(const HostParam &param) final;

    void Close() final;

    void Solve(const HostParam &param) final;

    const float *Fetch() final;

    float **compute_lstm_cell(int i, int cell_id, float **input_pp);

    float *W_dev[t_num_layer << 2], *U_dev[t_num_layer << 2],
        *bias_dev[t_num_layer << 2];
    float *state_c_dev[t_num_layer], *state_c_bak_dev[t_num_layer],
        *state_h_dev[t_num_layer];
    float *input_dev, *temp_output_dev[t_num_layer];

    float *output_host;
};

} // namespace mica::experiments::lstm

template <unsigned int hidden_size>
void OneKernel(const float *__restrict__ data,
               const float *__restrict__ state_c,
               const float *__restrict__ state_h, const float *__restrict__ W0,
               const float *__restrict__ W1, const float *__restrict__ W2,
               const float *__restrict__ W3, const float *__restrict__ U0,
               const float *__restrict__ U1, const float *__restrict__ U2,
               const float *__restrict__ U3, const float *__restrict__ bias0,
               const float *__restrict__ bias1, const float *__restrict__ bias2,
               const float *__restrict__ bias3, float *__restrict__ output,
               float *__restrict__ new_state);