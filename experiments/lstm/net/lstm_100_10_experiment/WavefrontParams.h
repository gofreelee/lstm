#pragma once
#define HIDDEN_SIZE 256
#include "cuda_runtime.h"

struct WaveInputParams {
    float *input_i;
    float *input_h;
};

struct WaveModelParams {
    float4 weight_w[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 weight_u[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 bias[HIDDEN_SIZE];
    
    float weight_ws[4][HIDDEN_SIZE * HIDDEN_SIZE];
    float weight_us[4][HIDDEN_SIZE * HIDDEN_SIZE];
    float biass[4][HIDDEN_SIZE];
    float temp[8][HIDDEN_SIZE];
};

struct WaveOutputParams {
    float4 *wi;
    float4 *uh;
    float *state_c;
    float *state_h;
};