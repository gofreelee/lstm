#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(512, 1)wave_compute_21(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_fusedcompute(21* LstmScaleParams::kCellNumber10 + 0, 0, 21* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 1:call_onekernel_compute_fusedcompute(20* LstmScaleParams::kCellNumber10 + 1, 1, 20* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 2:call_onekernel_compute_fusedcompute(19* LstmScaleParams::kCellNumber10 + 2, 2, 19* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 3:call_onekernel_compute_fusedcompute(18* LstmScaleParams::kCellNumber10 + 3, 3, 18* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 4:call_onekernel_compute_fusedcompute(17* LstmScaleParams::kCellNumber10 + 4, 4, 17* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 5:call_onekernel_compute_fusedcompute(16* LstmScaleParams::kCellNumber10 + 5, 5, 16* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 6:call_onekernel_compute_fusedcompute(15* LstmScaleParams::kCellNumber10 + 6, 6, 15* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 7:call_onekernel_compute_fusedcompute(14* LstmScaleParams::kCellNumber10 + 7, 7, 14* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 8:call_onekernel_compute_fusedcompute(13* LstmScaleParams::kCellNumber10 + 8, 8, 13* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;case 9:call_onekernel_compute_fusedcompute(12* LstmScaleParams::kCellNumber10 + 9, 9, 12* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, 512, LstmScaleParams::kMask7);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_21(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve_fusedcompute(21* LstmScaleParams::kCellNumber10 + 0, 0, 21* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_solve_fusedcompute(20* LstmScaleParams::kCellNumber10 + 1, 1, 20* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_solve_fusedcompute(19* LstmScaleParams::kCellNumber10 + 2, 2, 19* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_solve_fusedcompute(18* LstmScaleParams::kCellNumber10 + 3, 3, 18* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_solve_fusedcompute(17* LstmScaleParams::kCellNumber10 + 4, 4, 17* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_solve_fusedcompute(16* LstmScaleParams::kCellNumber10 + 5, 5, 16* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_solve_fusedcompute(15* LstmScaleParams::kCellNumber10 + 6, 6, 15* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 7:call_onekernel_solve_fusedcompute(14* LstmScaleParams::kCellNumber10 + 7, 7, 14* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 8:call_onekernel_solve_fusedcompute(13* LstmScaleParams::kCellNumber10 + 8, 8, 13* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 9:call_onekernel_solve_fusedcompute(12* LstmScaleParams::kCellNumber10 + 9, 9, 12* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}