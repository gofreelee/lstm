#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(512, 1)wave_compute_3(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_fusedcompute(3* LstmScaleParams::kCellNumber10 + 0, 0, 3* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 1:call_onekernel_compute_fusedcompute(2* LstmScaleParams::kCellNumber10 + 1, 1, 2* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 2:call_onekernel_compute_fusedcompute(1* LstmScaleParams::kCellNumber10 + 2, 2, 1* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 3:call_onekernel_compute_fusedcompute(0* LstmScaleParams::kCellNumber10 + 3, 3, 0* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_3(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve_fusedcompute(3* LstmScaleParams::kCellNumber10 + 0, 0, 3* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_solve_fusedcompute(2* LstmScaleParams::kCellNumber10 + 1, 1, 2* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_solve_fusedcompute(1* LstmScaleParams::kCellNumber10 + 2, 2, 1* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_solve_fusedcompute(0* LstmScaleParams::kCellNumber10 + 3, 3, 0* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}