#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(512, 1)wave_compute_84(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_fusedcompute(84* LstmScaleParams::kCellNumber10 + 0, 0, 84* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 1:call_onekernel_compute_fusedcompute(83* LstmScaleParams::kCellNumber10 + 1, 1, 83* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 2:call_onekernel_compute_fusedcompute(82* LstmScaleParams::kCellNumber10 + 2, 2, 82* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 3:call_onekernel_compute_fusedcompute(81* LstmScaleParams::kCellNumber10 + 3, 3, 81* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 4:call_onekernel_compute_fusedcompute(80* LstmScaleParams::kCellNumber10 + 4, 4, 80* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 5:call_onekernel_compute_fusedcompute(79* LstmScaleParams::kCellNumber10 + 5, 5, 79* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 6:call_onekernel_compute_fusedcompute(78* LstmScaleParams::kCellNumber10 + 6, 6, 78* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 7:call_onekernel_compute_fusedcompute(77* LstmScaleParams::kCellNumber10 + 7, 7, 77* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 8:call_onekernel_compute_fusedcompute(76* LstmScaleParams::kCellNumber10 + 8, 8, 76* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 9:call_onekernel_compute_fusedcompute(75* LstmScaleParams::kCellNumber10 + 9, 9, 75* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_84(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve_fusedcompute(84* LstmScaleParams::kCellNumber10 + 0, 0, 84* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_solve_fusedcompute(83* LstmScaleParams::kCellNumber10 + 1, 1, 83* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_solve_fusedcompute(82* LstmScaleParams::kCellNumber10 + 2, 2, 82* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_solve_fusedcompute(81* LstmScaleParams::kCellNumber10 + 3, 3, 81* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_solve_fusedcompute(80* LstmScaleParams::kCellNumber10 + 4, 4, 80* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_solve_fusedcompute(79* LstmScaleParams::kCellNumber10 + 5, 5, 79* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_solve_fusedcompute(78* LstmScaleParams::kCellNumber10 + 6, 6, 78* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 7:call_onekernel_solve_fusedcompute(77* LstmScaleParams::kCellNumber10 + 7, 7, 77* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 8:call_onekernel_solve_fusedcompute(76* LstmScaleParams::kCellNumber10 + 8, 8, 76* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 9:call_onekernel_solve_fusedcompute(75* LstmScaleParams::kCellNumber10 + 9, 9, 75* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}