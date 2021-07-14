#pragma once
#include <string_view>

extern "C" {

enum {
    kAlexnetImageWidth = 224,
    kAlexnetImageHeight = 224,
    kAlexnetImageChannel = 3,
    kAlexnetNumClass = 1000,
    kAlexnetBatchSize = 1,
};

struct AlexnetInput {
    float data[kAlexnetBatchSize * kAlexnetImageHeight * kAlexnetImageWidth *
               kAlexnetImageChannel];
};

struct AlexnetOutput {
    float data[kAlexnetBatchSize * kAlexnetNumClass];
};

struct AlexnetState {
    enum {
        kStateSize = 19834667,
    };
    float data[kStateSize];
};

const AlexnetOutput *AlexnetInfer(const AlexnetInput *input,
                                  AlexnetState *state);
};