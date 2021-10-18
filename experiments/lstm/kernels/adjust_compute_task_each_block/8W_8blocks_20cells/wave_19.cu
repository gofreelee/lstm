#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 4)
    wave_compute_19(WaveInputParams *__restrict__ input,
                    WaveModelParams *__restrict__ model,
                    WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_compute_fusedcompute(
            19 * 20 + 0, 0, 19 * 20 + 0, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_compute_fusedcompute(
            18 * 20 + 1, 1, 18 * 20 + 1, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_compute_fusedcompute(
            17 * 20 + 2, 2, 17 * 20 + 2, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 3:
        call_onekernel_compute_fusedcompute(
            16 * 20 + 3, 3, 16 * 20 + 3, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 4:
        call_onekernel_compute_fusedcompute(
            15 * 20 + 4, 4, 15 * 20 + 4, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 5:
        call_onekernel_compute_fusedcompute(
            14 * 20 + 5, 5, 14 * 20 + 5, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 6:
        call_onekernel_compute_fusedcompute(
            13 * 20 + 6, 6, 13 * 20 + 6, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 7:
        call_onekernel_compute_fusedcompute(
            12 * 20 + 7, 7, 12 * 20 + 7, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 8:
        call_onekernel_compute_fusedcompute(
            11 * 20 + 8, 8, 11 * 20 + 8, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 9:
        call_onekernel_compute_fusedcompute(
            10 * 20 + 9, 9, 10 * 20 + 9, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 10:
        call_onekernel_compute_fusedcompute(
            9 * 20 + 10, 10, 9 * 20 + 10, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 11:
        call_onekernel_compute_fusedcompute(
            8 * 20 + 11, 11, 8 * 20 + 11, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 12:
        call_onekernel_compute_fusedcompute(
            7 * 20 + 12, 12, 7 * 20 + 12, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 13:
        call_onekernel_compute_fusedcompute(
            6 * 20 + 13, 13, 6 * 20 + 13, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 14:
        call_onekernel_compute_fusedcompute(
            5 * 20 + 14, 14, 5 * 20 + 14, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 15:
        call_onekernel_compute_fusedcompute(
            4 * 20 + 15, 15, 4 * 20 + 15, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 16:
        call_onekernel_compute_fusedcompute(
            3 * 20 + 16, 16, 3 * 20 + 16, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 17:
        call_onekernel_compute_fusedcompute(
            2 * 20 + 17, 17, 2 * 20 + 17, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 18:
        call_onekernel_compute_fusedcompute(
            1 * 20 + 18, 18, 1 * 20 + 18, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 19:
        call_onekernel_compute_fusedcompute(
            0 * 20 + 19, 19, 0 * 20 + 19, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}
__global__ void __launch_bounds__(256, 4)
    wave_solve_19(WaveInputParams *__restrict__ input,
                  WaveModelParams *__restrict__ model,
                  WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_solve_fusedcompute(
            19 * 20 + 0, 0, 19 * 20 + 0, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_solve_fusedcompute(
            18 * 20 + 1, 1, 18 * 20 + 1, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_solve_fusedcompute(
            17 * 20 + 2, 2, 17 * 20 + 2, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 3:
        call_onekernel_solve_fusedcompute(
            16 * 20 + 3, 3, 16 * 20 + 3, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 4:
        call_onekernel_solve_fusedcompute(
            15 * 20 + 4, 4, 15 * 20 + 4, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 5:
        call_onekernel_solve_fusedcompute(
            14 * 20 + 5, 5, 14 * 20 + 5, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 6:
        call_onekernel_solve_fusedcompute(
            13 * 20 + 6, 6, 13 * 20 + 6, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 7:
        call_onekernel_solve_fusedcompute(
            12 * 20 + 7, 7, 12 * 20 + 7, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 8:
        call_onekernel_solve_fusedcompute(
            11 * 20 + 8, 8, 11 * 20 + 8, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 9:
        call_onekernel_solve_fusedcompute(
            10 * 20 + 9, 9, 10 * 20 + 9, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 10:
        call_onekernel_solve_fusedcompute(
            9 * 20 + 10, 10, 9 * 20 + 10, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 11:
        call_onekernel_solve_fusedcompute(
            8 * 20 + 11, 11, 8 * 20 + 11, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 12:
        call_onekernel_solve_fusedcompute(
            7 * 20 + 12, 12, 7 * 20 + 12, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 13:
        call_onekernel_solve_fusedcompute(
            6 * 20 + 13, 13, 6 * 20 + 13, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 14:
        call_onekernel_solve_fusedcompute(
            5 * 20 + 14, 14, 5 * 20 + 14, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 15:
        call_onekernel_solve_fusedcompute(
            4 * 20 + 15, 15, 4 * 20 + 15, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 16:
        call_onekernel_solve_fusedcompute(
            3 * 20 + 16, 16, 3 * 20 + 16, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 17:
        call_onekernel_solve_fusedcompute(
            2 * 20 + 17, 17, 2 * 20 + 17, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 18:
        call_onekernel_solve_fusedcompute(
            1 * 20 + 18, 18, 1 * 20 + 18, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 19:
        call_onekernel_solve_fusedcompute(
            0 * 20 + 19, 19, 0 * 20 + 19, LstmScaleParams::kColumsPerBlock32,
            LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}