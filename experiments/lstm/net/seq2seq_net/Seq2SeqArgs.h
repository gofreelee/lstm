#pragma once
#include "cuda_runtime.h"
#define HIDDEN_SIZE 128

struct WaveInputParams {
    float *input_i;
    float *input_h;
};

struct WaveModelParams {
    float4 weight_w[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 weight_u[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 bias[HIDDEN_SIZE];
};

struct WaveOutputParams {
    float4 *wi;
    float4 *uh;
    float *state_c;
    float *state_h;
};

struct LSTMNetHostParams {
    float *inputs;
    float *state_c_s;
    float *state_h_s;
    float *weights_w;
    float *weights_u;
    float *bias_s;
};