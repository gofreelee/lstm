#include "NaiveLstm.h"
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
    cudaMalloc(&sum_cached_ptr, sizeof(float) * ((t_hidden_size >> 5) << 2) *
                                    (t_hidden_size * 2));

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
    cudaStreamCreate(&stream_t);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmOneKernelNet<t_num_step, t_num_layer, t_batch_size,
                      t_hidden_size>::Close() {
    cudaFree(input_dev);
    free(output_host);
    cudaFree(sum_cached_ptr);
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
    cudaMemset(sum_cached_ptr, 0,
               sizeof(float) * ((t_hidden_size >> 5) << 2) * (t_hidden_size));
    void *args_x[] = {input_pp,
                      state_c_pp,
                      state_h_pp,
                      &W_dev[(cell_id << 2) + 0],
                      &U_dev[(cell_id << 2) + 0],
                      &bias_dev[(cell_id << 2) + 0],
                      &sum_cached_ptr};
    void *args_y[] = {input_pp,
                      state_c_pp,
                      state_h_pp,
                      &W_dev[(cell_id << 2) + 1],
                      &U_dev[(cell_id << 2) + 1],
                      &bias_dev[(cell_id << 2) + 1],
                      &sum_cached_ptr};
    void *args_z[] = {input_pp,
                      state_c_pp,
                      state_h_pp,
                      &W_dev[(cell_id << 2) + 2],
                      &U_dev[(cell_id << 2) + 2],
                      &bias_dev[(cell_id << 2) + 2],
                      &sum_cached_ptr};
    void *args_k[] = {input_pp,
                      state_c_pp,
                      state_h_pp,
                      &W_dev[(cell_id << 2) + 3],
                      &U_dev[(cell_id << 2) + 3],
                      &bias_dev[(cell_id << 2) + 3],
                      &sum_cached_ptr};
    void *args_solve[] = {state_c_pp, new_state_h_pp, new_state_c_pp,
                          &sum_cached_ptr};
    cudaLaunchKernel((const void *)compute_x<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args_x, 0, stream_t);
    cudaLaunchKernel((const void *)compute_y<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args_y, 0, stream_t);
    cudaLaunchKernel((const void *)compute_z<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args_z, 0, stream_t);
    cudaLaunchKernel((const void *)compute_k<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args_k, 0, stream_t);
    cudaLaunchKernel((const void *)solve<256>, dim3(2 * t_hidden_size),
                     dim3(t_hidden_size), (void **)args_solve, 0, stream_t);
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
        cudaMemset(sum_cached_ptr, 0,
                   sizeof(float) * ((t_hidden_size >> 5) << 2) *
                       (2 * t_hidden_size));
        void *args_x[] = {&input_dev, state_c_pp,   state_h_pp,     &W_dev[0],
                          &U_dev[0],  &bias_dev[0], &sum_cached_ptr};
        void *args_y[] = {&input_dev, state_c_pp,   state_h_pp,     &W_dev[1],
                          &U_dev[1],  &bias_dev[1], &sum_cached_ptr};
        void *args_z[] = {&input_dev, state_c_pp,   state_h_pp,     &W_dev[2],
                          &U_dev[2],  &bias_dev[2], &sum_cached_ptr};
        void *args_k[] = {&input_dev, state_c_pp,   state_h_pp,     &W_dev[3],
                          &U_dev[3],  &bias_dev[3], &sum_cached_ptr};
        void *args_solve[] = {state_c_pp, new_state_h_pp, new_state_c_pp,
                              &sum_cached_ptr};
        cudaLaunchKernel((const void *)compute_x<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args_x, 0, stream_t);
        cudaLaunchKernel((const void *)compute_y<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args_y, 0, stream_t);
        cudaLaunchKernel((const void *)compute_z<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args_z, 0, stream_t);
        cudaLaunchKernel((const void *)compute_k<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args_k, 0, stream_t);
        cudaLaunchKernel((const void *)solve<256>, dim3(2 * t_hidden_size),
                         dim3(t_hidden_size), (void **)args_solve, 0, stream_t);


        float **input_pp = new_state_h_pp;

        for (int j = 1; j < t_num_layer; ++j)
            input_pp = compute_lstm_cell(i, j, input_pp); // 输出作为输入
    }
}

template class LstmOneKernelNet<8, 8, 1, 256>;

} // namespace mica::experiments::lstm
