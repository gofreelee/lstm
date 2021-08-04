#pragma once
#define HIDDEN_SIZE 256
#include "cuda_runtime.h"

struct WaveInputParamsBS16 {
    float4 *input_i;
    float4 *input_h;
};

struct WaveModelParamsBS16 {
    float4 weight_w[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 weight_u[HIDDEN_SIZE * HIDDEN_SIZE];
    float4 bias[HIDDEN_SIZE];
};

struct WaveOutputParamsBS16 {
    float4 *wi;
    float4 *uh;
    float4 *state_c;
    float4 *state_h;
};