#pragma once
#define HIDDEN_SIZE 256
#include "cuda_runtime.h"

struct WaveInputParamsBS4 {
    float4 *input_i;
    float4 *input_h;
    float *input_i_f1;
    float *input_h_f1;
};

struct WaveModelParamsBS4 {
    float4 weight_w[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 weight_u[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 bias[HIDDEN_SIZE];

    float weight_ws[4][HIDDEN_SIZE * HIDDEN_SIZE];
    float weight_us[4][HIDDEN_SIZE * HIDDEN_SIZE];
    float biass[4][HIDDEN_SIZE];
    float temp[16][HIDDEN_SIZE];
};

struct WaveOutputParamsBS4 {
    float4 *wi;
    float4 *uh;
    float4 *state_c;
    float4 *state_h;
    float *state_c_f1;
    float *state_h_f1;
};