#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave_compute_6(WaveInputParams *__restrict__ input,
                   WaveModelParams *__restrict__ model,
                   WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_compute_naivefuse(
            6 * LstmScaleParams::kCellNumber10 + 0, 0,
            6 * LstmScaleParams::kCellNumber10 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_compute_naivefuse(
            5 * LstmScaleParams::kCellNumber10 + 1, 1,
            5 * LstmScaleParams::kCellNumber10 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_compute_naivefuse(
            4 * LstmScaleParams::kCellNumber10 + 2, 2,
            4 * LstmScaleParams::kCellNumber10 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 3:
        call_onekernel_compute_naivefuse(
            3 * LstmScaleParams::kCellNumber10 + 3, 3,
            3 * LstmScaleParams::kCellNumber10 + 3,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 4:
        call_onekernel_compute_naivefuse(
            2 * LstmScaleParams::kCellNumber10 + 4, 4,
            2 * LstmScaleParams::kCellNumber10 + 4,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 5:
        call_onekernel_compute_naivefuse(
            1 * LstmScaleParams::kCellNumber10 + 5, 5,
            1 * LstmScaleParams::kCellNumber10 + 5,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 6:
        call_onekernel_compute_naivefuse(
            0 * LstmScaleParams::kCellNumber10 + 6, 6,
            0 * LstmScaleParams::kCellNumber10 + 6,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave_solve_6(WaveInputParams *__restrict__ input,
                 WaveModelParams *__restrict__ model,
                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_solve_naivefuse(
            6 * LstmScaleParams::kCellNumber10 + 0, 0,
            6 * LstmScaleParams::kCellNumber10 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_solve_naivefuse(
            5 * LstmScaleParams::kCellNumber10 + 1, 1,
            5 * LstmScaleParams::kCellNumber10 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_solve_naivefuse(
            4 * LstmScaleParams::kCellNumber10 + 2, 2,
            4 * LstmScaleParams::kCellNumber10 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 3:
        call_onekernel_solve_naivefuse(
            3 * LstmScaleParams::kCellNumber10 + 3, 3,
            3 * LstmScaleParams::kCellNumber10 + 3,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 4:
        call_onekernel_solve_naivefuse(
            2 * LstmScaleParams::kCellNumber10 + 4, 4,
            2 * LstmScaleParams::kCellNumber10 + 4,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 5:
        call_onekernel_solve_naivefuse(
            1 * LstmScaleParams::kCellNumber10 + 5, 5,
            1 * LstmScaleParams::kCellNumber10 + 5,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 6:
        call_onekernel_solve_naivefuse(
            0 * LstmScaleParams::kCellNumber10 + 6, 6,
            0 * LstmScaleParams::kCellNumber10 + 6,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}