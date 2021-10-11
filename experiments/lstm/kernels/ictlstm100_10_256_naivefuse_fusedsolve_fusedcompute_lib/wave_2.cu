#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave2(WaveInputParams *__restrict__ input,
          WaveModelParams *__restrict__ model,
          WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_naivefuse_fusedsolve_fusedcompute(
            2 * LstmScaleParams::kCellNumber10 + 0, 0,
            2 * LstmScaleParams::kCellNumber10 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_naivefuse_fusedsolve_fusedcompute(
            1 * LstmScaleParams::kCellNumber10 + 1, 1,
            1 * LstmScaleParams::kCellNumber10 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_naivefuse_fusedsolve_fusedcompute(
            0 * LstmScaleParams::kCellNumber10 + 2, 2,
            0 * LstmScaleParams::kCellNumber10 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}