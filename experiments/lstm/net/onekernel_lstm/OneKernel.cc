#include "OneKernel.h"
#include "cuda_runtime.h"
#include <iostream>

namespace mica::experiments::lstm {

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
const float *LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                              t_hidden_size>::Fetch() {
    if (!hasFetchOutput) {
        cudaMemcpy(output_host,
                   num_step & 0x1 ? temp_output_dev[t_num_layer - 1]
                                  : state_h_dev[t_num_layer - 1],
                   sizeof(float) * t_batch_size * t_hidden_size,
                   cudaMemcpyDeviceToHost);
        hasFetchOutput = true;
    }
    return output_host;
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                      t_hidden_size>::Initialize(const HostParam &param) {
    cudaMalloc(&input_dev, sizeof(float) * t_hidden_size * t_batch_size *
                                   (4 * t_num_layer + 1) +
                               sizeof(float) *
                                   (t_hidden_size * t_hidden_size * 2 +
                                    t_hidden_size * t_batch_size) *
                                   4 * t_num_layer);
    cudaMemcpy(input_dev, param.input[0],
               sizeof(float) * t_hidden_size * t_batch_size,
               cudaMemcpyHostToDevice);

    for (auto i = 0; i < t_num_layer; ++i) {
        temp_output_dev[i] = input_dev + t_hidden_size * t_batch_size +
                             i * t_hidden_size * t_batch_size * 4;
        state_c_dev[i] = temp_output_dev[i] + t_hidden_size * t_batch_size;
        state_c_bak_dev[i] = state_c_dev[i] + t_hidden_size * t_batch_size;
        state_h_dev[i] = state_c_bak_dev[i] + t_hidden_size * t_batch_size;

        cudaMemcpy(state_c_dev[i], param.init_state[i],
                   sizeof(float) * t_hidden_size * t_batch_size,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(state_h_dev[i],
                   param.init_state[i] + t_hidden_size * t_batch_size,
                   sizeof(float) * t_hidden_size * t_batch_size,
                   cudaMemcpyHostToDevice);
    }
    //查看第一个元素的初始c 和 h

    for (auto i = 0; i < t_num_layer; ++i) {
        for (int j = 0; j < 4; ++j) {
            W_dev[(i << 2) + j] =
                input_dev +
                ((1 + 4 * t_num_layer) * t_hidden_size * t_batch_size +
                 i * 4 *
                     (t_hidden_size * t_hidden_size * 2 +
                      t_hidden_size * t_batch_size) +
                 j * (t_hidden_size * t_hidden_size * 2 +
                      t_hidden_size * t_batch_size));
            U_dev[(i << 2) + j] =
                W_dev[(i << 2) + j] + t_hidden_size * t_hidden_size;
            bias_dev[(i << 2) + j] =
                U_dev[(i << 2) + j] + t_hidden_size * t_hidden_size;

            cudaMemcpy(W_dev[(i << 2) + j], param.W[(i << 2) + j],
                       sizeof(float) * t_hidden_size * t_hidden_size,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(U_dev[(i << 2) + j], param.U[(i << 2) + j],
                       sizeof(float) * t_hidden_size * t_hidden_size,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(bias_dev[(i << 2) + j], param.bias[(i << 2) + j],
                       sizeof(float) * t_batch_size * t_hidden_size,
                       cudaMemcpyHostToDevice);
        }
    }

    output_host = (float *)malloc(sizeof(float) * t_hidden_size * t_batch_size);
    cudaMemcpy(output_host, input_dev,
               sizeof(float) * t_hidden_size * t_batch_size,
               cudaMemcpyDeviceToHost);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                      t_hidden_size>::Close() {
    cudaFree(input_dev);
    free(output_host);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
float **LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                         t_hidden_size>::compute_lstm_cell(int i, int cell_id,
                                                           float **input_pp) {

    float **state_c_pp =
        (i & 0x1) ? &state_c_bak_dev[cell_id] : &state_c_dev[cell_id];
    float **state_h_pp =
        (i & 0x1) ? &temp_output_dev[cell_id] : &state_h_dev[cell_id];
    float **new_state_c_pp =
        (i & 0x1) ? &state_c_dev[cell_id] : &state_c_bak_dev[cell_id];
    float **new_state_h_pp =
        (i & 0x1) ? &state_h_dev[cell_id] : &temp_output_dev[cell_id];
    void *args[] = {input_pp,
                    state_c_pp,
                    state_h_pp,
                    &W_dev[(cell_id << 2) + 0],
                    &W_dev[(cell_id << 2) + 1],
                    &W_dev[(cell_id << 2) + 2],
                    &W_dev[(cell_id << 2) + 3],
                    &U_dev[(cell_id << 2) + 0],
                    &U_dev[(cell_id << 2) + 1],
                    &U_dev[(cell_id << 2) + 2],
                    &U_dev[(cell_id << 2) + 3],
                    &bias_dev[(cell_id << 2) + 0],
                    &bias_dev[(cell_id << 2) + 1],
                    &bias_dev[(cell_id << 2) + 2],
                    &bias_dev[(cell_id << 2) + 3],
                    new_state_h_pp,
                    new_state_c_pp};
    cudaLaunchKernel((const void *)OneKernel<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args,
                     sizeof(float) * ((t_hidden_size >> 5) << 2));
    return new_state_h_pp;
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                      t_hidden_size>::Solve(const HostParam &param) {

    hasFetchOutput = false;
    for (auto i = 0; i < num_step; ++i) {
        float **state_c_pp = (i & 0x1) ? &state_c_bak_dev[0] : &state_c_dev[0];
        float **state_h_pp = (i & 0x1) ? &temp_output_dev[0] : &state_h_dev[0];
        float **new_state_c_pp =
            (i & 0x1) ? &state_c_dev[0] : &state_c_bak_dev[0];
        float **new_state_h_pp =
            (i & 0x1) ? &state_h_dev[0] : &temp_output_dev[0];
        void *args[] = {&input_dev,     state_c_pp,    state_h_pp,
                        &W_dev[0],      &W_dev[1],     &W_dev[2],
                        &W_dev[3],      &U_dev[0],     &U_dev[1],
                        &U_dev[2],      &U_dev[3],     &bias_dev[0],
                        &bias_dev[1],   &bias_dev[2],  &bias_dev[3],
                        new_state_h_pp, new_state_c_pp};
        cudaMemcpy(output_host, input_dev,
                   sizeof(float) * t_hidden_size * t_batch_size,
                   cudaMemcpyDeviceToHost);

        cudaLaunchKernel((const void *)OneKernel<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args,
                         sizeof(float) * ((t_hidden_size >> 5) << 2));
        // O -> O   O  O  O O  O 0

        float **input_pp = new_state_h_pp;

        for (int j = 1; j < t_num_layer; ++j)
            input_pp = compute_lstm_cell(i, j, input_pp); // 输出作为输入
    }
}

template class LstmOneKernelNet<8, 8, 1, 256>;

} // namespace mica::experiments::lstm
