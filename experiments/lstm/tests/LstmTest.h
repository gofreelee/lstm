#pragma once

#include "gtest/gtest.h"
#include <cmath>
#include <iostream>

namespace mica::experiments::lstm {

namespace unittest {

class LstmTest : public ::testing::Test {
  public:
    static float W_0_0[], U_0_0[], bias_0_0[], W_0_1[], U_0_1[], bias_0_1[],
        W_0_2[], U_0_2[], bias_0_2[], W_0_3[], U_0_3[], bias_0_3[], W_1_0[],
        U_1_0[], bias_1_0[], W_1_1[], U_1_1[], bias_1_1[], W_1_2[], U_1_2[],
        bias_1_2[], W_1_3[], U_1_3[], bias_1_3[], W_2_0[], U_2_0[], bias_2_0[],
        W_2_1[], U_2_1[], bias_2_1[], W_2_2[], U_2_2[], bias_2_2[], W_2_3[],
        U_2_3[], bias_2_3[], W_3_0[], U_3_0[], bias_3_0[], W_3_1[], U_3_1[],
        bias_3_1[], W_3_2[], U_3_2[], bias_3_2[], W_3_3[], U_3_3[], bias_3_3[],
        W_4_0[], U_4_0[], bias_4_0[], W_4_1[], U_4_1[], bias_4_1[], W_4_2[],
        U_4_2[], bias_4_2[], W_4_3[], U_4_3[], bias_4_3[], W_5_0[], U_5_0[],
        bias_5_0[], W_5_1[], U_5_1[], bias_5_1[], W_5_2[], U_5_2[], bias_5_2[],
        W_5_3[], U_5_3[], bias_5_3[], W_6_0[], U_6_0[], bias_6_0[], W_6_1[],
        U_6_1[], bias_6_1[], W_6_2[], U_6_2[], bias_6_2[], W_6_3[], U_6_3[],
        bias_6_3[], W_7_0[], U_7_0[], bias_7_0[], W_7_1[], U_7_1[], bias_7_1[],
        W_7_2[], U_7_2[], bias_7_2[], W_7_3[], U_7_3[], bias_7_3[], kExpected[];

  protected:
    inline float diff(float x, float y) {
        float diff = fabs(x - y);
        return diff > 0.00001f ? diff : 0.0f;
    }

    void ASSERT_OUTPUT(const float *output) {
        for (int i = 0; i < 256; ++i) {
            // std::cout << output[i] << " " << kExpected[i] << std::endl;
            ASSERT_FLOAT_EQ(diff(output[i], kExpected[i]), 0.0f);
        }
    }
};

} // namespace unittest
} // namespace mica::experiments::lstm
