#pragma once

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

namespace mica::experiments::lstm {

namespace unittest {

class LstmTest : public ::testing::Test {
  public:
    static float W_0_0[], U_0_0[], bias_0_0[], W_0_1[], U_0_1[], bias_0_1[],
        W_0_2[], U_0_2[], bias_0_2[], W_0_3[], U_0_3[], bias_0_3[];

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
