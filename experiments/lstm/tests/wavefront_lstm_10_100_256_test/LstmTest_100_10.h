#pragma once

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

namespace mica::experiments::lstm {

namespace unittest {

class LstmTest_100_10 : public ::testing::Test {

  protected:
    inline float diff(float x, float y) {
        float diff = fabs(x - y);
        return diff > 0.00001f ? diff : 0.0f;
    }

    void ASSERT_OUTPUT(const float *output, const float *kExpected) {
        for (int i = 0; i < 256; ++i) {
            ASSERT_FLOAT_EQ(diff(output[i], kExpected[i]), 0.0f);
        }
    }
};

} // namespace unittest
} // namespace mica::experiments::lstm
