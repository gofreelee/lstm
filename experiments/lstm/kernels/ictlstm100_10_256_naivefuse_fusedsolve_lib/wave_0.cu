#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave0(WaveInputParams *__restrict__ input,
          WaveModelParams *__restrict__ model,
          WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_naivefuse_fusedsolve(
            0 * LstmScaleParams::kCellNumber10 + 0, 0,
            0 * LstmScaleParams::kCellNumber10 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}