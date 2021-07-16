#include "include/RammerLike2.h"
#include "include/KernelUtils.h"
#include "include/LstmCommon.h"
#include "include/RammerLikeArgs.h"

#include "cuda_runtime.h"
#include <cstring>
#include <iostream>

namespace mica::experiments::lstm {

void MatrixMulVector(float t_hidden_size, const float matrix[],
                     const float vector[], float output[]) {
    for (int col = 0; col < t_hidden_size; ++col) {
        output[col] = 0.0000f;
        for (int addr = col, row = 0; row < t_hidden_size;
             ++row, addr += t_hidden_size)
            output[col] += matrix[addr] * vector[row];
    }
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
const float *LstmRammerLike2Net<t_num_step, t_num_layer, t_batch_size,
                                t_hidden_size>::getOutput() {
    if (!hasFetchOutput) {
        cudaMemcpy(output_host, state_h_dev[(t_num_step + 1) * t_num_layer - 1],
                   sizeof(float) * t_batch_size * t_hidden_size,
                   cudaMemcpyDeviceToHost);
        hasFetchOutput = true;
    }
    return reinterpret_cast<float *>(output_host);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmRammerLike2Net<t_num_step, t_num_layer, t_batch_size,
                        t_hidden_size>::init(const float *input[],
                                             const float *init_state[],
                                             const float *W[], const float *U[],
                                             const float *bias[]) {

    cudaMalloc(&input_dev, sizeof(float) * t_hidden_size * t_batch_size);
    cudaMemcpy(input_dev, input[0],
               sizeof(float) * t_hidden_size * t_batch_size,
               cudaMemcpyHostToDevice);

    for (int step = 0; step <= t_num_step; ++step) {
        for (int cell = 0; cell < t_num_layer; ++cell) {
            if (step != t_num_step)
                cudaMalloc(&state_c_dev[step * t_num_layer + cell],
                           sizeof(float) * t_hidden_size * t_batch_size);
            cudaMalloc(&state_h_dev[step * t_num_layer + cell],
                       sizeof(float) * t_hidden_size * t_batch_size);

            if (step == 0) {
                cudaMemcpy(state_c_dev[cell], init_state[cell],
                           sizeof(float) * t_hidden_size * t_batch_size,
                           cudaMemcpyHostToDevice);

                cudaMemcpy(state_h_dev[cell], init_state[cell],
                           sizeof(float) * t_hidden_size * t_batch_size,
                           cudaMemcpyHostToDevice);
            }
        }
    }

    cudaMalloc(&models,
               sizeof(RammerLikeCellModel<t_hidden_size>) * t_num_layer);
    for (int cell = 0; cell < t_num_layer; ++cell) {
        RammerLikeCellModel<t_hidden_size> *model = &models[cell];
        for (int i = 0; i < 4; ++i) {
            cudaMemcpy(&model->W[i][0], W[(cell << 2) + i],
                       sizeof(float) * t_hidden_size * t_hidden_size,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&model->U[i][0], U[(cell << 2) + i],
                       sizeof(float) * t_hidden_size * t_hidden_size,
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&model->bias[i][0], bias[(cell << 2) + i],
                       sizeof(float) * t_hidden_size, cudaMemcpyHostToDevice);
        }
    }

    RammerLikeCellOutput cell_outputs_host[t_num_layer * t_num_step];
    auto *cell_inputs_host = (RammerLikeCellInput<t_hidden_size> *)malloc(
        sizeof(RammerLikeCellInput<t_hidden_size>) * t_num_layer * t_num_step);
    for (int step = 0; step < t_num_step; ++step) {
        for (int cell = 0; cell < t_num_layer; ++cell) {
            cell_inputs_host[step * t_num_layer + cell].data =
                cell == 0
                    ? input_dev
                    : state_h_dev[(step + 1) * t_num_layer + cell - 1]; // ?
            cell_inputs_host[step * t_num_layer + cell].state_c =
                state_c_dev[step * t_num_layer + cell];
            cell_inputs_host[step * t_num_layer + cell].state_h =
                state_h_dev[step * t_num_layer + cell];

            if (step == 0) {
                for (int i = 0; i < 4; ++i) {
                    MatrixMulVector(t_hidden_size, U[(cell << 2) + i],
                                    init_state[0],
                                    &cell_inputs_host[cell]
                                         .UMulStateHResult[i * t_hidden_size]);
                }
            }

            cell_outputs_host[step * t_num_layer + cell].new_state_c =
                state_c_dev[(step + 1) * t_num_layer + cell];
            cell_outputs_host[step * t_num_layer + cell].new_state_h =
                state_h_dev[(step + 1) * t_num_layer + cell];
        }
    }
    cudaMalloc(&cell_inputs, sizeof(RammerLikeCellInput<t_hidden_size>) *
                                 t_num_layer * t_num_step);
    cudaMemcpy(cell_inputs, cell_inputs_host,
               sizeof(RammerLikeCellInput<t_hidden_size>) * t_num_layer *
                   t_num_step,
               cudaMemcpyHostToDevice);
    delete cell_inputs_host;

    cudaMalloc(&cell_outputs,
               sizeof(RammerLikeCellOutput) * t_num_layer * t_num_step);
    cudaMemcpy(cell_outputs, cell_outputs_host,
               sizeof(RammerLikeCellOutput) * t_num_layer * t_num_step,
               cudaMemcpyHostToDevice);

    output_host =
        (float4 *)malloc(sizeof(float) * t_hidden_size * t_batch_size);

    cudaStreamCreate(&stream);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmRammerLike2Net<t_num_step, t_num_layer, t_batch_size,
                        t_hidden_size>::finalize() {
    free(output_host);
    cudaFree(input_dev);
    cudaFree(models);
    cudaFree(cell_inputs);
    cudaFree(cell_outputs);
    for (int step = 0; step <= t_num_step; ++step) {
        for (int cell = 0; cell < t_num_layer; ++cell) {
            if (step != t_num_step)
                cudaFree(state_c_dev[step * t_num_layer + cell]);
            cudaFree(state_h_dev[step * t_num_layer + cell]);
        }
    }

    cudaStreamDestroy(stream);
}

template <size_t t_num_layer, size_t t_hidden_size>
static void launchRammerLikeKernels(RammerLikeCellInput<t_hidden_size> *inputs,
                                    RammerLikeCellModel<t_hidden_size> *models,
                                    RammerLikeCellOutput *outputs,
                                    cudaStream_t stream) {

    void *args[] = {&inputs, &models, &outputs};
    cudaLaunchKernel((void *)ok_1<t_hidden_size, t_num_layer>, dim3(256),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok0<t_hidden_size, t_num_layer>, dim3(1),
                     dim3(256), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok1<t_hidden_size, t_num_layer>, dim3(16),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok2<t_hidden_size, t_num_layer>, dim3(24),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok3<t_hidden_size, t_num_layer>, dim3(32),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok4<t_hidden_size, t_num_layer>, dim3(40),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok5<t_hidden_size, t_num_layer>, dim3(48),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok6<t_hidden_size, t_num_layer>, dim3(56),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok7<t_hidden_size, t_num_layer>, dim3(64),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok8<t_hidden_size, t_num_layer>, dim3(56),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok9<t_hidden_size, t_num_layer>, dim3(48),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok10<t_hidden_size, t_num_layer>, dim3(40),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok11<t_hidden_size, t_num_layer>, dim3(32),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok12<t_hidden_size, t_num_layer>, dim3(24),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok13<t_hidden_size, t_num_layer>, dim3(16),
                     dim3(128), (void **)args, 0, stream);
    cudaLaunchKernel((void *)ok14<t_hidden_size, t_num_layer>, dim3(8),
                     dim3(128), (void **)args, 0, stream);
}

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
void LstmRammerLike2Net<t_num_step, t_num_layer, t_batch_size,
                        t_hidden_size>::compute(const float *input[],
                                                const float *init_state[],
                                                const float *W[],
                                                const float *U[],
                                                const float *bias[]) {

    hasFetchOutput = false;
    static_assert((t_num_step & 0x1) == 0 && (t_num_layer & 0x1) == 0);
    static_assert(t_num_layer == t_num_step);
    launchRammerLikeKernels<t_num_layer, t_hidden_size>(cell_inputs, models,
                                                        cell_outputs, stream);
}

template class LstmRammerLike2Net<8, 8, 1, 256>;

} // namespace mica::experiments::lstm