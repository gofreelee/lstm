#pragma once
#include <string_view>

extern "C" {

enum {
    kResnet50ImageWidth = 224,
    kResnet50ImageHeight = 224,
    kResnet50ImageChannel = 3,
    kResnet50NumClass = 1000,
    kResnet50BatchSize = 1,

    kResnext29ImageWidth = 32,
    kResnext29ImageHeight = 32,
    kResnext29ImageChannel = 3,
    kResnext29NumClass = 10,
    kResnext29BatchSize = 1,
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

struct Resnext29Input {
    float data[kResnext29BatchSize * kResnext29ImageHeight *
               kResnext29ImageWidth * kResnext29ImageChannel];
};

struct Resnext29Output {
    float data[kResnext29BatchSize * kResnext29NumClass];
};

struct Resnext29State {
    enum {
        kStateSize = 141521876,
    };
    float data[kStateSize];
};

const Resnext29Output *Resnext29Infer(const Resnext29Input *input,
                                      Resnext29State *state);
};