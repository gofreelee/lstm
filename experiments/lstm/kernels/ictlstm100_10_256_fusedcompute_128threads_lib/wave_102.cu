#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave_compute_102(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_fusedcompute(99* LstmScaleParams::kCellNumber10 + 3, 3, 99* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 1:call_onekernel_compute_fusedcompute(98* LstmScaleParams::kCellNumber10 + 4, 4, 98* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 2:call_onekernel_compute_fusedcompute(97* LstmScaleParams::kCellNumber10 + 5, 5, 97* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 3:call_onekernel_compute_fusedcompute(96* LstmScaleParams::kCellNumber10 + 6, 6, 96* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 4:call_onekernel_compute_fusedcompute(95* LstmScaleParams::kCellNumber10 + 7, 7, 95* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 5:call_onekernel_compute_fusedcompute(94* LstmScaleParams::kCellNumber10 + 8, 8, 94* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;case 6:call_onekernel_compute_fusedcompute(93* LstmScaleParams::kCellNumber10 + 9, 9, 93* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock128, LstmScaleParams::kMask7);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_102(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve_fusedcompute(99* LstmScaleParams::kCellNumber10 + 3, 3, 99* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_solve_fusedcompute(98* LstmScaleParams::kCellNumber10 + 4, 4, 98* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_solve_fusedcompute(97* LstmScaleParams::kCellNumber10 + 5, 5, 97* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_solve_fusedcompute(96* LstmScaleParams::kCellNumber10 + 6, 6, 96* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_solve_fusedcompute(95* LstmScaleParams::kCellNumber10 + 7, 7, 95* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_solve_fusedcompute(94* LstmScaleParams::kCellNumber10 + 8, 8, 94* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_solve_fusedcompute(93* LstmScaleParams::kCellNumber10 + 9, 9, 93* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}