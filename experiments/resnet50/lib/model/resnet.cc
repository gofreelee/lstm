#include "resnet.h"
#include "op/avg_pool.h"
#include "op/bottleneck_add.h"
#include "op/conv.h"
#include "op/linear.h"
#include "op/max_pool.h"
#include "op/utils.h"
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>

template <uint32_t kBatchSize, uint32_t kHeight, uint32_t kWidth,
          uint32_t kChannelIn, uint32_t kChannelBase, bool kNeedDownsample,
          uint32_t kStride, uint32_t kGroupSize, uint32_t kWidthPerGroup>
class BasicBlock {
  public:
    enum {
        kInputSize = kBatchSize * kChannelIn * kHeight * kWidth,
        kChannelOut = 4 * kChannelBase,
        kChannelConv1 = kChannelBase * kWidthPerGroup / 64 * kGroupSize,
        kHeightOut = kHeight / kStride,
        kWidthOut = kWidth / kStride,
        kSize = kBatchSize * kChannelOut * kHeightOut * kWidthOut,
        kNormalGroupSize = 1,

        kConv1FilterHW = 1,
        kConv1Padding = 0,
        kConv1StrideHW = 1,

        kConv2FilterHW = 3,
        kConv2Padding = 1,
        kConv2StrideHW = kStride,

        kConv3FilterHW = 1,
        kConv3Padding = 0,
        kConv3StrideHW = 1,

        kConvDownsampleFilterHW = 1,
        kConvDownsamplePadding = 0,
        kConvDownsampleStrideHW = kStride,

    };

    typedef struct {
        typename StateTraits<
            Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelConv1,
                 kConv1FilterHW, kConv1FilterHW, kConv1Padding, kConv1Padding,
                 kConv1StrideHW, kConv1StrideHW, false, true, true,
                 kNormalGroupSize>>::StateType fuse_conv1_bn_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kHeight, kWidth, kChannelConv1, kChannelConv1,
                 kConv2FilterHW, kConv2FilterHW, kConv2Padding, kConv2Padding,
                 kConv2StrideHW, kConv2StrideHW, false, true, true,
                 kGroupSize>>::StateType fuse_conv2_bn_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kHeightOut, kWidthOut, kChannelConv1, kChannelOut,
                 kConv3FilterHW, kConv3FilterHW, kConv3Padding, kConv3Padding,
                 kConv3StrideHW, kConv3StrideHW, false, true, false,
                 kNormalGroupSize>>::StateType fuse_conv3_bn_state;
        typename StateTraits<
            Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelOut,
                 kConvDownsampleFilterHW, kConvDownsampleFilterHW,
                 kConvDownsamplePadding, kConvDownsamplePadding,
                 kConvDownsampleStrideHW, kConvDownsampleStrideHW, false, true,
                 false, kNormalGroupSize>>::StateType downsample_conv_bn_state;
        typename StateTraits<BottleneckAdd<
            kBatchSize, kBatchSize * kChannelOut * kHeightOut * kWidthOut,
            true>>::StateType fuse_add_relu_state;
    } StateWithDownsample;

    typedef struct {
        typename StateTraits<
            Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelConv1,
                 kConv1FilterHW, kConv1FilterHW, kConv1Padding, kConv1Padding,
                 kConv1StrideHW, kConv1StrideHW, false, true, true,
                 kNormalGroupSize>>::StateType fuse_conv1_bn_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kHeight, kWidth, kChannelConv1, kChannelConv1,
                 kConv2FilterHW, kConv2FilterHW, kConv2Padding, kConv2Padding,
                 kConv2StrideHW, kConv2StrideHW, false, true, true,
                 kGroupSize>>::StateType fuse_conv2_bn_relu_state;
        typename StateTraits<
            Conv<kBatchSize, kHeightOut, kWidthOut, kChannelConv1, kChannelOut,
                 kConv3FilterHW, kConv3FilterHW, kConv3Padding, kConv3Padding,
                 kConv3StrideHW, kConv3StrideHW, false, true, false,
                 kNormalGroupSize>>::StateType fuse_conv3_bn_state;
        typename StateTraits<BottleneckAdd<
            kBatchSize, kBatchSize * kChannelOut * kHeightOut * kWidthOut,
            true>>::StateType fuse_add_relu_state;
    } StateWithoutDownsample;

    typedef typename std::conditional<kNeedDownsample, StateWithDownsample,
                                      StateWithoutDownsample>::type StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelConv1,
             kConv1FilterHW, kConv1FilterHW, kConv1Padding, kConv1Padding,
             kConv1StrideHW, kConv1StrideHW, false, true, true,
             kNormalGroupSize>::Forward(input,
                                        &state->fuse_conv1_bn_relu_state);

        Conv<kBatchSize, kHeight, kWidth, kChannelConv1, kChannelConv1,
             kConv2FilterHW, kConv2FilterHW, kConv2Padding, kConv2Padding,
             kConv2StrideHW, kConv2StrideHW, false, true, true,
             kGroupSize>::Forward(&state->fuse_conv1_bn_relu_state.output,
                                  &state->fuse_conv2_bn_relu_state);

        Conv<kBatchSize, kHeightOut, kWidthOut, kChannelConv1, kChannelOut,
             kConv3FilterHW, kConv3FilterHW, kConv3Padding, kConv3Padding,
             kConv3StrideHW, kConv3StrideHW, false, true, false,
             kNormalGroupSize>::Forward(&state->fuse_conv2_bn_relu_state.output,
                                        &state->fuse_conv3_bn_state);

        if constexpr (kNeedDownsample) {
            Conv<kBatchSize, kHeight, kWidth, kChannelIn, kChannelOut,
                 kConvDownsampleFilterHW, kConvDownsampleFilterHW,
                 kConvDownsamplePadding, kConvDownsamplePadding,
                 kConvDownsampleStrideHW, kConvDownsampleStrideHW, false, true,
                 false,
                 kNormalGroupSize>::Forward(input,
                                            &state->downsample_conv_bn_state);
            BottleneckAdd<
                kBatchSize, kBatchSize * kChannelOut * kHeightOut * kWidthOut,
                true>::Forward(&state->downsample_conv_bn_state.output,
                               &state->fuse_conv3_bn_state.output,
                               &state->fuse_add_relu_state);
        } else {
            BottleneckAdd<kBatchSize,
                          kBatchSize * kChannelOut * kHeightOut * kWidthOut,
                          true>::Forward(input,
                                         &state->fuse_conv3_bn_state.output,
                                         &state->fuse_add_relu_state);
        }
    }
};

template <uint32_t kBatchSize, uint32_t kHeight, uint32_t kWidth,
          uint32_t kChannelIn, uint32_t kChannelBase, uint32_t kRepeatNumber,
          uint32_t kStride, uint32_t kGroupSize, uint32_t kWidthPerGroup>
class BottleNeck {
  public:
    enum {
        kBottleNeckExpansion = 4,
        kInputSize = kBatchSize * kChannelIn * kHeight * kWidth,
        kChannelOut = kBottleNeckExpansion * kChannelBase,
        kHeightOut = kHeight / kStride,
        kWidthOut = kWidth / kStride,
        kSize = kBatchSize * kChannelOut * kHeightOut * kWidthOut,
        kRepeatBasicBlockStride = 1,
    };

    typedef struct {
        typename StateTraits<
            BasicBlock<kBatchSize, kHeight, kWidth, kChannelIn, kChannelBase,
                       true, kStride, kGroupSize, kWidthPerGroup>>::StateType
            block1_state;

        typename StateTraits<BasicBlock<
            kBatchSize, kHeightOut, kWidthOut, kChannelOut, kChannelBase, false,
            kRepeatBasicBlockStride, kGroupSize, kWidthPerGroup>>::StateType
            blocks_state[kRepeatNumber - 1];
    } StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        BasicBlock<kBatchSize, kHeight, kWidth, kChannelIn, kChannelBase, true,
                   kStride, kGroupSize,
                   kWidthPerGroup>::Forward(input, &state->block1_state);

        for (unsigned i = 0; i < kRepeatNumber - 1; ++i) {
            if (i == 0) {
                BasicBlock<kBatchSize, kHeightOut, kWidthOut, kChannelOut,
                           kChannelBase, false, kRepeatBasicBlockStride,
                           kGroupSize,
                           kWidthPerGroup>::Forward(&state->block1_state
                                                         .fuse_add_relu_state,
                                                    &state->blocks_state[0]);
            } else {
                BasicBlock<kBatchSize, kHeightOut, kWidthOut, kChannelOut,
                           kChannelBase, false, kRepeatBasicBlockStride,
                           kGroupSize,
                           kWidthPerGroup>::Forward(&state->blocks_state[i - 1]
                                                         .fuse_add_relu_state,
                                                    &state->blocks_state[i]);
            }
        }
    }
};

template <uint32_t kBatchSize, uint32_t kNumClass> class Resnet50Model {
  public:
    enum {
        kSize = kBatchSize * kNumClass,
        kInputSize = kBatchSize * kResnet50ImageChannel * kResnet50ImageHeight *
                     kResnet50ImageWidth,
        kBottleNeckExpansion = 4,
        kGroupSize = 1,
        kWidthPerGroup = 64,

        kConv1FilterHW = 7,
        kConv1Padding = 3,
        kConv1StrideHW = 2,
        kConv1ChannelOut = 64,
        kConv1OutputHW = kResnet50ImageHeight / kConv1StrideHW,

        kMaxpool1InputHW = kConv1OutputHW,
        kMaxpool1ChannelIn = kConv1ChannelOut,
        kMaxpool1FilterHW = 3,
        kMaxpool1Padding = 1,
        kMaxpool1StrideHW = 2,
        kMaxpool1ChannelOut = kMaxpool1ChannelIn,
        kMaxpool1OutputHW = kMaxpool1InputHW / kMaxpool1StrideHW,

        kConv2BottleNeckInputHW = kMaxpool1OutputHW,
        kConv2BottleNeckChannelIn = kMaxpool1ChannelOut,
        kConv2BottleNeckChannelBase = kConv2BottleNeckChannelIn,
        kConv2BottleNeckRepeat = 3,
        kConv2BottleNeckStride = 1,
        kConv2BottleNeckChannelOut =
            kConv2BottleNeckChannelBase * kBottleNeckExpansion,
        kConv2BottleNeckOutputHW =
            kConv2BottleNeckInputHW / kConv2BottleNeckStride,

        kConv3BottleNeckInputHW = kConv2BottleNeckOutputHW,
        kConv3BottleNeckChannelIn = kConv2BottleNeckChannelOut,
        kConv3BottleNeckChannelBase = kConv2BottleNeckChannelBase * 2,
        kConv3BottleNeckRepeat = 4,
        kConv3BottleNeckStride = 2,
        kConv3BottleNeckChannelOut =
            kConv3BottleNeckChannelBase * kBottleNeckExpansion,
        kConv3BottleNeckOutputHW =
            kConv3BottleNeckInputHW / kConv3BottleNeckStride,

        kConv4BottleNeckInputHW = kConv3BottleNeckOutputHW,
        kConv4BottleNeckChannelIn = kConv3BottleNeckChannelOut,
        kConv4BottleNeckChannelBase = kConv3BottleNeckChannelBase * 2,
        kConv4BottleNeckRepeat = 6,
        kConv4BottleNeckStride = 2,
        kConv4BottleNeckChannelOut =
            kConv4BottleNeckChannelBase * kBottleNeckExpansion,
        kConv4BottleNeckOutputHW =
            kConv4BottleNeckInputHW / kConv4BottleNeckStride,

        kConv5BottleNeckInputHW = kConv4BottleNeckOutputHW,
        kConv5BottleNeckChannelIn = kConv4BottleNeckChannelOut,
        kConv5BottleNeckChannelBase = kConv4BottleNeckChannelBase * 2,
        kConv5BottleNeckRepeat = 3,
        kConv5BottleNeckStride = 2,
        kConv5BottleNeckChannelOut =
            kConv5BottleNeckChannelBase * kBottleNeckExpansion,
        kConv5BottleNeckOutputHW =
            kConv5BottleNeckInputHW / kConv5BottleNeckStride,

        kAvgpool1InputHW = kConv5BottleNeckOutputHW,
        kAvgpool1ChannelIn = kConv5BottleNeckChannelOut,
        kAvgpool1FilterHW = kAvgpool1InputHW,
        kAvgpool1Padding = 0,
        kAvgpool1StrideHW = kAvgpool1FilterHW,
        kAvgpool1ChannelOut = kAvgpool1ChannelIn,
        kAvgpool1OutputHW = kAvgpool1InputHW / kAvgpool1StrideHW,

        kLinear1InputSize =
            kAvgpool1ChannelOut * kAvgpool1OutputHW * kAvgpool1OutputHW,

    };

    typedef struct {
        typename StateTraits<
            Conv<kBatchSize, kResnet50ImageHeight, kResnet50ImageWidth,
                 kResnet50ImageChannel, kConv1ChannelOut, kConv1FilterHW,
                 kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
                 kConv1StrideHW, false, true, true, 1>>::StateType
            fuse_conv1_bn_relu_state;
        typename StateTraits<MaxPool<
            kBatchSize, kMaxpool1ChannelIn, kMaxpool1InputHW, kConv1OutputHW,
            kMaxpool1FilterHW, kMaxpool1FilterHW, kMaxpool1Padding,
            kMaxpool1Padding, kMaxpool1StrideHW, kMaxpool1StrideHW>>::StateType
            maxpool1_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv2BottleNeckInputHW, kConv2BottleNeckInputHW,
            kConv2BottleNeckChannelIn, kConv2BottleNeckChannelBase,
            kConv2BottleNeckRepeat, kConv2BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv2_bottleneck_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv3BottleNeckInputHW, kConv3BottleNeckInputHW,
            kConv3BottleNeckChannelIn, kConv3BottleNeckChannelBase,
            kConv3BottleNeckRepeat, kConv3BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv3_bottleneck_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv4BottleNeckInputHW, kConv4BottleNeckInputHW,
            kConv4BottleNeckChannelIn, kConv4BottleNeckChannelBase,
            kConv4BottleNeckRepeat, kConv4BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv4_bottleneck_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv5BottleNeckInputHW, kConv5BottleNeckInputHW,
            kConv5BottleNeckChannelIn, kConv5BottleNeckChannelBase,
            kConv5BottleNeckRepeat, kConv5BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv5_bottleneck_state;
        typename StateTraits<AvgPool<
            kBatchSize, kAvgpool1ChannelIn, kAvgpool1InputHW, kAvgpool1InputHW,
            kAvgpool1FilterHW, kAvgpool1FilterHW, kAvgpool1Padding,
            kAvgpool1Padding, kAvgpool1StrideHW, kAvgpool1StrideHW>>::StateType
            avgpool1_state;
        typename StateTraits<Linear<kBatchSize, kLinear1InputSize, kNumClass,
                                    true, false>>::StateType fc1_state;
    } StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        Conv<kBatchSize, kResnet50ImageHeight, kResnet50ImageWidth,
             kResnet50ImageChannel, kConv1ChannelOut, kConv1FilterHW,
             kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
             kConv1StrideHW, false, true, true,
             1>::Forward(input, &state->fuse_conv1_bn_relu_state);

        MaxPool<kBatchSize, kMaxpool1ChannelIn, kMaxpool1InputHW,
                kConv1OutputHW, kMaxpool1FilterHW, kMaxpool1FilterHW,
                kMaxpool1Padding, kMaxpool1Padding, kMaxpool1StrideHW,
                kMaxpool1StrideHW>::Forward(&state->fuse_conv1_bn_relu_state
                                                 .output,
                                            &state->maxpool1_state);

        BottleNeck<kBatchSize, kConv2BottleNeckInputHW, kConv2BottleNeckInputHW,
                   kConv2BottleNeckChannelIn, kConv2BottleNeckChannelBase,
                   kConv2BottleNeckRepeat, kConv2BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::Forward(&state->maxpool1_state.output,
                                            &state->conv2_bottleneck_state);

        BottleNeck<kBatchSize, kConv3BottleNeckInputHW, kConv3BottleNeckInputHW,
                   kConv3BottleNeckChannelIn, kConv3BottleNeckChannelBase,
                   kConv3BottleNeckRepeat, kConv3BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::
            Forward(&state->conv2_bottleneck_state
                         .blocks_state[kConv2BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->conv3_bottleneck_state);

        BottleNeck<kBatchSize, kConv4BottleNeckInputHW, kConv4BottleNeckInputHW,
                   kConv4BottleNeckChannelIn, kConv4BottleNeckChannelBase,
                   kConv4BottleNeckRepeat, kConv4BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::
            Forward(&state->conv3_bottleneck_state
                         .blocks_state[kConv3BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->conv4_bottleneck_state);

        BottleNeck<kBatchSize, kConv5BottleNeckInputHW, kConv5BottleNeckInputHW,
                   kConv5BottleNeckChannelIn, kConv5BottleNeckChannelBase,
                   kConv5BottleNeckRepeat, kConv5BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::
            Forward(&state->conv4_bottleneck_state
                         .blocks_state[kConv4BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->conv5_bottleneck_state);

        AvgPool<kBatchSize, kAvgpool1ChannelIn, kAvgpool1InputHW,
                kAvgpool1InputHW, kAvgpool1FilterHW, kAvgpool1FilterHW,
                kAvgpool1Padding, kAvgpool1Padding, kAvgpool1StrideHW,
                kAvgpool1StrideHW>::
            Forward(&state->conv5_bottleneck_state
                         .blocks_state[kConv5BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->avgpool1_state);

        Linear<kBatchSize, kLinear1InputSize, kNumClass, true, false>::Forward(
            &state->avgpool1_state.output, &state->fc1_state);
    }
};

template <uint32_t kBatchSize, uint32_t kNumClass> class Resnext29Model {
  public:
    enum {
        kSize = kBatchSize * kNumClass,
        kInputSize = kBatchSize * kResnext29ImageChannel *
                     kResnext29ImageHeight * kResnext29ImageWidth,
        kBottleNeckExpansion = 4,
        kGroupSize = 16,
        kWidthPerGroup = 64,

        kConv1FilterHW = 3,
        kConv1Padding = 1,
        kConv1StrideHW = 1,
        kConv1ChannelOut = 64,
        kConv1OutputHW = kResnext29ImageHeight / kConv1StrideHW,

        kConv2BottleNeckInputHW = kConv1OutputHW,
        kConv2BottleNeckChannelIn = kConv1ChannelOut,
        kConv2BottleNeckChannelBase = kConv2BottleNeckChannelIn,
        kConv2BottleNeckRepeat = 3,
        kConv2BottleNeckStride = 1,
        kConv2BottleNeckChannelOut =
            kConv2BottleNeckChannelBase * kBottleNeckExpansion,
        kConv2BottleNeckOutputHW =
            kConv2BottleNeckInputHW / kConv2BottleNeckStride,

        kConv3BottleNeckInputHW = kConv2BottleNeckOutputHW,
        kConv3BottleNeckChannelIn = kConv2BottleNeckChannelOut,
        kConv3BottleNeckChannelBase = kConv2BottleNeckChannelBase * 2,
        kConv3BottleNeckRepeat = 3,
        kConv3BottleNeckStride = 2,
        kConv3BottleNeckChannelOut =
            kConv3BottleNeckChannelBase * kBottleNeckExpansion,
        kConv3BottleNeckOutputHW =
            kConv3BottleNeckInputHW / kConv3BottleNeckStride,

        kConv4BottleNeckInputHW = kConv3BottleNeckOutputHW,
        kConv4BottleNeckChannelIn = kConv3BottleNeckChannelOut,
        kConv4BottleNeckChannelBase = kConv3BottleNeckChannelBase * 2,
        kConv4BottleNeckRepeat = 3,
        kConv4BottleNeckStride = 2,
        kConv4BottleNeckChannelOut =
            kConv4BottleNeckChannelBase * kBottleNeckExpansion,
        kConv4BottleNeckOutputHW =
            kConv4BottleNeckInputHW / kConv4BottleNeckStride,

        kAvgpool1InputHW = kConv4BottleNeckOutputHW,
        kAvgpool1ChannelIn = kConv4BottleNeckChannelOut,
        kAvgpool1FilterHW = kAvgpool1InputHW,
        kAvgpool1Padding = 0,
        kAvgpool1StrideHW = kAvgpool1FilterHW,
        kAvgpool1ChannelOut = kAvgpool1ChannelIn,
        kAvgpool1OutputHW = kAvgpool1InputHW / kAvgpool1StrideHW,

        kLinear1InputSize =
            kAvgpool1ChannelOut * kAvgpool1OutputHW * kAvgpool1OutputHW,

    };

    typedef struct {
        typename StateTraits<
            Conv<kBatchSize, kResnext29ImageHeight, kResnext29ImageWidth,
                 kResnext29ImageChannel, kConv1ChannelOut, kConv1FilterHW,
                 kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
                 kConv1StrideHW, false, true, true, 1>>::StateType
            fuse_conv1_bn_relu_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv2BottleNeckInputHW, kConv2BottleNeckInputHW,
            kConv2BottleNeckChannelIn, kConv2BottleNeckChannelBase,
            kConv2BottleNeckRepeat, kConv2BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv2_bottleneck_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv3BottleNeckInputHW, kConv3BottleNeckInputHW,
            kConv3BottleNeckChannelIn, kConv3BottleNeckChannelBase,
            kConv3BottleNeckRepeat, kConv3BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv3_bottleneck_state;
        typename StateTraits<BottleNeck<
            kBatchSize, kConv4BottleNeckInputHW, kConv4BottleNeckInputHW,
            kConv4BottleNeckChannelIn, kConv4BottleNeckChannelBase,
            kConv4BottleNeckRepeat, kConv4BottleNeckStride, kGroupSize,
            kWidthPerGroup>>::StateType conv4_bottleneck_state;
        typename StateTraits<AvgPool<
            kBatchSize, kAvgpool1ChannelIn, kAvgpool1InputHW, kAvgpool1InputHW,
            kAvgpool1FilterHW, kAvgpool1FilterHW, kAvgpool1Padding,
            kAvgpool1Padding, kAvgpool1StrideHW, kAvgpool1StrideHW>>::StateType
            avgpool1_state;
        typename StateTraits<Linear<kBatchSize, kLinear1InputSize, kNumClass,
                                    true, false>>::StateType fc1_state;
    } StateType;

    static void Forward(const Tensor<kInputSize> *input, StateType *state) {

        Conv<kBatchSize, kResnext29ImageHeight, kResnext29ImageWidth,
             kResnext29ImageChannel, kConv1ChannelOut, kConv1FilterHW,
             kConv1FilterHW, kConv1Padding, kConv1Padding, kConv1StrideHW,
             kConv1StrideHW, false, true, true,
             1>::Forward(input, &state->fuse_conv1_bn_relu_state);

        BottleNeck<kBatchSize, kConv2BottleNeckInputHW, kConv2BottleNeckInputHW,
                   kConv2BottleNeckChannelIn, kConv2BottleNeckChannelBase,
                   kConv2BottleNeckRepeat, kConv2BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::Forward(&state->fuse_conv1_bn_relu_state
                                                 .output,
                                            &state->conv2_bottleneck_state);

        BottleNeck<kBatchSize, kConv3BottleNeckInputHW, kConv3BottleNeckInputHW,
                   kConv3BottleNeckChannelIn, kConv3BottleNeckChannelBase,
                   kConv3BottleNeckRepeat, kConv3BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::
            Forward(&state->conv2_bottleneck_state
                         .blocks_state[kConv2BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->conv3_bottleneck_state);

        BottleNeck<kBatchSize, kConv4BottleNeckInputHW, kConv4BottleNeckInputHW,
                   kConv4BottleNeckChannelIn, kConv4BottleNeckChannelBase,
                   kConv4BottleNeckRepeat, kConv4BottleNeckStride, kGroupSize,
                   kWidthPerGroup>::
            Forward(&state->conv3_bottleneck_state
                         .blocks_state[kConv3BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->conv4_bottleneck_state);

        AvgPool<kBatchSize, kAvgpool1ChannelIn, kAvgpool1InputHW,
                kAvgpool1InputHW, kAvgpool1FilterHW, kAvgpool1FilterHW,
                kAvgpool1Padding, kAvgpool1Padding, kAvgpool1StrideHW,
                kAvgpool1StrideHW>::
            Forward(&state->conv4_bottleneck_state
                         .blocks_state[kConv4BottleNeckRepeat - 2]
                         .fuse_add_relu_state,
                    &state->avgpool1_state);

        Linear<kBatchSize, kLinear1InputSize, kNumClass, true, false>::Forward(
            &state->avgpool1_state.output, &state->fc1_state);
    }
};

const Resnet50Output *Resnet50Infer(const Resnet50Input *input,
                                    Resnet50State *state) {
    using InputType = Tensor<
        Resnet50Model<kResnet50BatchSize, kResnet50NumClass>::kInputSize>;
    using StateType = StateTraits<
        Resnet50Model<kResnet50BatchSize, kResnet50NumClass>>::StateType;
    static_assert(sizeof(*state) == sizeof(StateType));
    static_assert(
        sizeof(reinterpret_cast<StateType *>(state)->fc1_state.output) ==
        sizeof(Resnet50Output));
    Resnet50Model<kResnet50BatchSize, kResnet50NumClass>::Forward(
        reinterpret_cast<const InputType *>(input),
        reinterpret_cast<StateType *>(state));

    return reinterpret_cast<Resnet50Output *>(
        &reinterpret_cast<StateType *>(state)->fc1_state.output);
}

const Resnext29Output *Resnext29Infer(const Resnext29Input *input,
                                      Resnext29State *state) {
    using InputType = Tensor<
        Resnext29Model<kResnext29BatchSize, kResnext29NumClass>::kInputSize>;
    using StateType = StateTraits<
        Resnext29Model<kResnext29BatchSize, kResnext29NumClass>>::StateType;
    static_assert(sizeof(*state) == sizeof(StateType));
    static_assert(
        sizeof(reinterpret_cast<StateType *>(state)->fc1_state.output) ==
        sizeof(Resnext29Output));
    Resnext29Model<kResnext29BatchSize, kResnext29NumClass>::Forward(
        reinterpret_cast<const InputType *>(input),
        reinterpret_cast<StateType *>(state));

    return reinterpret_cast<Resnext29Output *>(
        &reinterpret_cast<StateType *>(state)->fc1_state.output);
}
