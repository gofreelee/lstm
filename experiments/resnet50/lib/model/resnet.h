#pragma once
#include <string_view>

extern "C" {

enum {
    kResnet50ImageWidth = 224,
    kResnet50ImageHeight = 224,
    kResnet50ImageChannel = 3,
    kResnet50NumClass = 1000,
    kResnet50BatchSize = 1,
};

struct Resnet50Input {
    float data[kResnet50BatchSize * kResnet50ImageHeight * kResnet50ImageWidth *
               kResnet50ImageChannel];
};

struct Resnet50Output {
    float data[kResnet50BatchSize * kResnet50NumClass];
};

struct Resnet50State {
    enum {
        kStateSize = 64464016,
    };
    float data[kStateSize];
};

const Resnet50Output *Resnet50Infer(const Resnet50Input *input,
                                    Resnet50State *state);
};