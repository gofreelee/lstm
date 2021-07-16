#ifndef _RAMMERLIKE_ARG_H_
#define _RAMMERLIKE_ARG_H_

#include "cuda_runtime.h"

template <unsigned int t_hidden_size> struct RammerLikeCellInput {
    const float4 *data;
    const float4 *state_c;
    const float4 *state_h;
    float4 *WMulData;

    float WMulDataResult[4 * t_hidden_size];
    float UMulStateHResult[4 * t_hidden_size];
};

struct RammerLikeCellOutput {
    float4 *new_state_h;
    float4 *new_state_c;
};

template <unsigned int t_hidden_size> struct RammerLikeCellModel {
    float4 W[4][t_hidden_size * (t_hidden_size >> 2)];
    float4 U[4][t_hidden_size * (t_hidden_size >> 2)];
    float bias[4][t_hidden_size];
};

#endif