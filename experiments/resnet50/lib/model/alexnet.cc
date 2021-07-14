#include "alexnet.h"
#include "op/conv.h"
#include "op/linear.h"
#include "op/max_pool.h"
#include "op/utils.h"
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <string_view>

template <uint32_t kBatchSize, uint32_t kNumClass> class AlexnetModel {
  public:
    enum {
        kInputSize = kBatchSize * kAlexnetImageChannel * kAlexnetImageHeight *
                     kAlexnetImageWidth,
        kSize = kBatchSize * kNumClass,

        kConv1FilterHW = 11,
        kConv1Padding = 2,
        kConv1StrideHW = 4,
        kConv1ChannelOut = 48,
        kConv1OutputHW =
            (kAlexnetImageHeight + 2 * kConv1Padding - kConv1FilterHW) /
                kConv1StrideHW +
            1,

        kMaxpool1InputHW = kConv1OutputHW,
        kMaxpool1ChannelIn = kConv1ChannelOut,
        kMaxpool1FilterHW = 3,
        kMaxpool1Padding = 0,
        kMaxpool1StrideHW = 2,
        kMaxpool1ChannelOut = kMaxpool1ChannelIn,
        kMaxpool1OutputHW =
            (kMaxpool1InputHW + 2 * kMaxpool1Padding - kMaxpool1FilterHW) /
                kMaxpool1StrideHW +
            1,

        kConv2InputHW = kMaxpool1OutputHW,
        kConv2ChannelIn = kMaxpool1ChannelOut,
        kConv2FilterHW = 5,
        kConv2Padding = 2,
        kConv2StrideHW = 1,
        kConv2ChannelOut = 128,
        kConv2OutputHW = (kConv2InputHW + 2 * kConv2Padding - kConv2FilterHW) /
                             kConv2StrideHW +
                         1,

        kMaxpool2InputHW = kConv2OutputHW,
        kMaxpool2ChannelIn = kConv2ChannelOut,
        kMaxpool2FilterHW = 3,
        kMaxpool2Padding = 0,
        kMaxpool2StrideHW = 2,
        kMaxpool2ChannelOut = kMaxpool2ChannelIn,
        kMaxpool2OutputHW =
            (kMaxpool2InputHW + 2 * kMaxpool2Padding - kMaxpool2FilterHW) /
                kMaxpool2StrideHW +
            1,

        kConv3InputHW = kMaxpool2OutputHW,
        kConv3ChannelIn = kMaxpool2ChannelOut,
        kConv3FilterHW = 3,
        kConv3Padding = 1,
        kConv3StrideHW = 1,
        kConv3ChannelOut = 192,
        kConv3OutputHW = (kConv3InputHW + 2 * kConv3Padding - kConv3FilterHW) /
                             kConv3StrideHW +
                         1,

        kConv4InputHW = kConv3OutputHW,
        kConv4ChannelIn = kConv3ChannelOut,
        kConv4FilterHW = 3,
        kConv4Padding = 1,
        kConv4StrideHW = 1,
        kConv4ChannelOut = 192,
        kConv4OutputHW = (kConv4InputHW + 2 * kConv4Padding - kConv4FilterHW) /
                             kConv4StrideHW +
                         1,

        kConv5InputHW = kConv4OutputHW,
        kConv5ChannelIn = kConv4ChannelOut,
        kConv5FilterHW = 3,
        kConv5Padding = 1,
        kConv5StrideHW = 1,
        kConv5ChannelOut = 128,
        kConv5OutputHW = (kConv5InputHW + 2 * kConv5Padding - kConv5FilterHW) /
                             kConv5StrideHW +
                         1,

        kMaxpool3InputHW = kConv5OutputHW,
        kMaxpool3ChannelIn = kConv5ChannelOut,
        kMaxpool3FilterHW = 3,
        kMaxpool3Padding = 0,
        kMaxpool3StrideHW = 2,
        kMaxpool3ChannelOut = kMaxpool3ChannelIn,
        kMaxpool3OutputHW =
            (kMaxpool3InputHW + 2 * kMaxpool3Padding - kMaxpool3FilterHW) /
                kMaxpool3StrideHW +
            1,

        kLinear1InputSize =
            kMaxpool3ChannelOut * kMaxpool3OutputHW * kMaxpool3OutputHW,
        kLinear1OutputSize = 2048,

        kLinear2InputSize = kLinear1OutputSize,
        kLinear2OutputSize = kLinear2InputSize,

        kLinear3InputSize = kLinear2OutputSize,
    };
    typedef struct {
        typename StateTraits<
            Conv<kBatchSize, kAlexnetImageHeight, kAlexnetImageHeight,
                 kAlexnetImageChannel, kConv1ChannelOut, kConv1FilterHW,
                 kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
                 kConv1StrideHW, true, false, true>>::StateType
            fuse_conv1_bias_relu_state;
        typename StateTraits<MaxPool<
            kBatchSize, kMaxpool1ChannelIn, kMaxpool1InputHW, kMaxpool1InputHW,
            kMaxpool1FilterHW, kMaxpool1FilterHW, kMaxpool1Padding,
            kMaxpool1Padding, kMaxpool1StrideHW, kMaxpool1StrideHW>>::StateType
            maxpool1_state;
        typename StateTraits<
            Conv<kBatchSize, kConv2InputHW, kConv2InputHW, kConv2ChannelIn,
                 kConv2ChannelOut, kConv2FilterHW, kConv2FilterHW,
                 kConv2Padding, kConv2Padding, kConv2StrideHW, kConv2StrideHW,
                 true, false, true>>::StateType fuse_conv2_bias_relu_state;
        typename StateTraits<MaxPool<
            kBatchSize, kMaxpool2ChannelIn, kMaxpool2InputHW, kMaxpool2InputHW,
            kMaxpool2FilterHW, kMaxpool2FilterHW, kMaxpool2Padding,
            kMaxpool2Padding, kMaxpool2StrideHW, kMaxpool2StrideHW>>::StateType
            maxpool2_state;
        typename StateTraits<
            Conv<kBatchSize, kConv3InputHW, kConv3InputHW, kConv3ChannelIn,
                 kConv3ChannelOut, kConv3FilterHW, kConv3FilterHW,
                 kConv3Padding, kConv3Padding, kConv3StrideHW, kConv3StrideHW,
                 true, false, true>>::StateType fuse_conv3_bias_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kConv4InputHW, kConv4InputHW, kConv4ChannelIn,
                 kConv4ChannelOut, kConv4FilterHW, kConv4FilterHW,
                 kConv4Padding, kConv4Padding, kConv4StrideHW, kConv4StrideHW,
                 true, false, true>>::StateType fuse_conv4_bias_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kConv5InputHW, kConv5InputHW, kConv5ChannelIn,
                 kConv5ChannelOut, kConv5FilterHW, kConv5FilterHW,
                 kConv5Padding, kConv5Padding, kConv5StrideHW, kConv5StrideHW,
                 true, false, true>>::StateType fuse_conv5_bias_relu_state;
        typename StateTraits<MaxPool<
            kBatchSize, kMaxpool3ChannelIn, kMaxpool3InputHW, kMaxpool3InputHW,
            kMaxpool3FilterHW, kMaxpool3FilterHW, kMaxpool3Padding,
            kMaxpool3Padding, kMaxpool3StrideHW, kMaxpool3StrideHW>>::StateType
            maxpool3_state;
        typename StateTraits<Linear<kBatchSize, kLinear1InputSize,
                                    kLinear1OutputSize, true, true>>::StateType
            fuse_fc1_relu_state;
        typename StateTraits<Linear<kBatchSize, kLinear2InputSize,
                                    kLinear2OutputSize, true, true>>::StateType
            fuse_fc2_relu_state;
        typename StateTraits<Linear<kBatchSize, kLinear3InputSize, kNumClass,
                                    true, false>>::StateType fc3_state;
    } StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        Conv<kBatchSize, kAlexnetImageHeight, kAlexnetImageHeight,
             kAlexnetImageChannel, kConv1ChannelOut, kConv1FilterHW,
             kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
             kConv1StrideHW, true, false,
             true>::Forward(input, &state->fuse_conv1_bias_relu_state);
        MaxPool<kBatchSize, kMaxpool1ChannelIn, kMaxpool1InputHW,
                kMaxpool1InputHW, kMaxpool1FilterHW, kMaxpool1FilterHW,
                kMaxpool1Padding, kMaxpool1Padding, kMaxpool1StrideHW,
                kMaxpool1StrideHW>::Forward(&state->fuse_conv1_bias_relu_state
                                                 .output,
                                            &state->maxpool1_state);
        Conv<kBatchSize, kConv2InputHW, kConv2InputHW, kConv2ChannelIn,
             kConv2ChannelOut, kConv2FilterHW, kConv2FilterHW, kConv2Padding,
             kConv2Padding, kConv2StrideHW, kConv2StrideHW, true, false,
             true>::Forward(&state->maxpool1_state.output,
                            &state->fuse_conv2_bias_relu_state);
        MaxPool<kBatchSize, kMaxpool2ChannelIn, kMaxpool2InputHW,
                kMaxpool2InputHW, kMaxpool2FilterHW, kMaxpool2FilterHW,
                kMaxpool2Padding, kMaxpool2Padding, kMaxpool2StrideHW,
                kMaxpool2StrideHW>::Forward(&state->fuse_conv2_bias_relu_state
                                                 .output,
                                            &state->maxpool2_state);
        Conv<kBatchSize, kConv3InputHW, kConv3InputHW, kConv3ChannelIn,
             kConv3ChannelOut, kConv3FilterHW, kConv3FilterHW, kConv3Padding,
             kConv3Padding, kConv3StrideHW, kConv3StrideHW, true, false,
             true>::Forward(&state->maxpool2_state.output,
                            &state->fuse_conv3_bias_relu_state);
        Conv<kBatchSize, kConv4InputHW, kConv4InputHW, kConv4ChannelIn,
             kConv4ChannelOut, kConv4FilterHW, kConv4FilterHW, kConv4Padding,
             kConv4Padding, kConv4StrideHW, kConv4StrideHW, true, false,
             true>::Forward(&state->fuse_conv3_bias_relu_state.output,
                            &state->fuse_conv4_bias_relu_state);
        Conv<kBatchSize, kConv5InputHW, kConv5InputHW, kConv5ChannelIn,
             kConv5ChannelOut, kConv5FilterHW, kConv5FilterHW, kConv5Padding,
             kConv5Padding, kConv5StrideHW, kConv5StrideHW, true, false,
             true>::Forward(&state->fuse_conv4_bias_relu_state.output,
                            &state->fuse_conv5_bias_relu_state);
        MaxPool<kBatchSize, kMaxpool3ChannelIn, kMaxpool3InputHW,
                kMaxpool3InputHW, kMaxpool3FilterHW, kMaxpool3FilterHW,
                kMaxpool3Padding, kMaxpool3Padding, kMaxpool3StrideHW,
                kMaxpool3StrideHW>::Forward(&state->fuse_conv5_bias_relu_state
                                                 .output,
                                            &state->maxpool3_state);
        Linear<kBatchSize, kLinear1InputSize, kLinear1OutputSize, true,
               true>::Forward(&state->maxpool3_state.output,
                              &state->fuse_fc1_relu_state);
        Linear<kBatchSize, kLinear2InputSize, kLinear2OutputSize, true,
               true>::Forward(&state->fuse_fc1_relu_state.output,
                              &state->fuse_fc2_relu_state);
        Linear<kBatchSize, kLinear3InputSize, kNumClass, true, false>::Forward(
            &state->fuse_fc2_relu_state.output, &state->fc3_state);
    }
};

const AlexnetOutput *AlexnetInfer(const AlexnetInput *input,
                                  AlexnetState *state) {
    using InputType =
        Tensor<AlexnetModel<kAlexnetBatchSize, kAlexnetNumClass>::kInputSize>;
    using StateType = StateTraits<
        AlexnetModel<kAlexnetBatchSize, kAlexnetNumClass>>::StateType;
    static_assert(sizeof(*state) == sizeof(StateType));
    static_assert(
        sizeof(reinterpret_cast<StateType *>(state)->fc3_state.output) ==
        sizeof(AlexnetOutput));
    AlexnetModel<kAlexnetBatchSize, kAlexnetNumClass>::Forward(
        reinterpret_cast<const InputType *>(input),
        reinterpret_cast<StateType *>(state));

    return reinterpret_cast<AlexnetOutput *>(
        &reinterpret_cast<StateType *>(state)->fc3_state.output);
}
