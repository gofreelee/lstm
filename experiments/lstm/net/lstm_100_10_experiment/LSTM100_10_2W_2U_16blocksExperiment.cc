#include "LSTM100_10_2W_2U_16blocksExperiment.h"
#include "WavefrontFunctionArgs.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mica::experiments::lstm {
void LSTM100_10_2W_2U_16blocksExperiment::computeAndSolve() {
    void *arg_s[] = {&waveInputParams_dev, &waveModelParams_dev,
                     &waveOutputParams_dev};
    cudaLaunchKernel((void *)wave_compute_0, dim3(16), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_0, dim3(8), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_1, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_1, dim3(16), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_2, dim3(48), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_2, dim3(24), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_3, dim3(64), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_3, dim3(32), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_4, dim3(80), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_4, dim3(40), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_5, dim3(96), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_5, dim3(48), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_6, dim3(112), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_6, dim3(56), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_7, dim3(128), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_7, dim3(64), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_8, dim3(144), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_8, dim3(72), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_9, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_9, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_10, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_10, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_11, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_11, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_12, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_12, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_13, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_13, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_14, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_14, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_15, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_15, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_16, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_16, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_17, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_17, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_18, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_18, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_19, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_19, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_20, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_20, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_21, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_21, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_22, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_22, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_23, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_23, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_24, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_24, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_25, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_25, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_26, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_26, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_27, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_27, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_28, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_28, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_29, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_29, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_30, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_30, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_31, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_31, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_32, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_32, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_33, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_33, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_34, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_34, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_35, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_35, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_36, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_36, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_37, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_37, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_38, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_38, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_39, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_39, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_40, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_40, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_41, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_41, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_42, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_42, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_43, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_43, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_44, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_44, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_45, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_45, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_46, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_46, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_47, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_47, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_48, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_48, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_49, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_49, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_50, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_50, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_51, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_51, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_52, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_52, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_53, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_53, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_54, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_54, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_55, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_55, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_56, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_56, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_57, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_57, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_58, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_58, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_59, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_59, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_60, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_60, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_61, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_61, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_62, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_62, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_63, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_63, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_64, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_64, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_65, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_65, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_66, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_66, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_67, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_67, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_68, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_68, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_69, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_69, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_70, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_70, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_71, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_71, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_72, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_72, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_73, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_73, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_74, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_74, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_75, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_75, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_76, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_76, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_77, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_77, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_78, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_78, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_79, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_79, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_80, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_80, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_81, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_81, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_82, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_82, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_83, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_83, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_84, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_84, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_85, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_85, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_86, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_86, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_87, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_87, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_88, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_88, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_89, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_89, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_90, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_90, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_91, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_91, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_92, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_92, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_93, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_93, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_94, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_94, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_95, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_95, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_96, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_96, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_97, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_97, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_98, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_98, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_99, dim3(160), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_99, dim3(80), dim3(256), (void **)arg_s,
                     0, stream);
    cudaLaunchKernel((void *)wave_compute_100, dim3(144), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_100, dim3(72), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_101, dim3(128), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_101, dim3(64), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_102, dim3(112), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_102, dim3(56), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_103, dim3(96), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_103, dim3(48), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_104, dim3(80), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_104, dim3(40), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_105, dim3(64), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_105, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_106, dim3(48), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_106, dim3(24), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_107, dim3(32), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_107, dim3(16), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_compute_108, dim3(16), dim3(256),
                     (void **)arg_s, 0, stream);
    cudaLaunchKernel((void *)wave_solve_108, dim3(8), dim3(256), (void **)arg_s,
                     0, stream);

    cudaDeviceSynchronize();
}
} // namespace mica::experiments::lstm