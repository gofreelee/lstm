#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 4) wave_compute_8(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_fusedcompute(8* LstmScaleParams::kCellNumber10 + 0, 0, 8* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_compute_fusedcompute(7* LstmScaleParams::kCellNumber10 + 1, 1, 7* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_compute_fusedcompute(6* LstmScaleParams::kCellNumber10 + 2, 2, 6* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_compute_fusedcompute(5* LstmScaleParams::kCellNumber10 + 3, 3, 5* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_compute_fusedcompute(4* LstmScaleParams::kCellNumber10 + 4, 4, 4* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_compute_fusedcompute(3* LstmScaleParams::kCellNumber10 + 5, 5, 3* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_compute_fusedcompute(2* LstmScaleParams::kCellNumber10 + 6, 6, 2* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 7:call_onekernel_compute_fusedcompute(1* LstmScaleParams::kCellNumber10 + 7, 7, 1* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 8:call_onekernel_compute_fusedcompute(0* LstmScaleParams::kCellNumber10 + 8, 8, 0* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_8(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve_fusedcompute(8* LstmScaleParams::kCellNumber10 + 0, 0, 8* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_solve_fusedcompute(7* LstmScaleParams::kCellNumber10 + 1, 1, 7* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_solve_fusedcompute(6* LstmScaleParams::kCellNumber10 + 2, 2, 6* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_solve_fusedcompute(5* LstmScaleParams::kCellNumber10 + 3, 3, 5* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_solve_fusedcompute(4* LstmScaleParams::kCellNumber10 + 4, 4, 4* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_solve_fusedcompute(3* LstmScaleParams::kCellNumber10 + 5, 5, 3* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_solve_fusedcompute(2* LstmScaleParams::kCellNumber10 + 6, 6, 2* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 7:call_onekernel_solve_fusedcompute(1* LstmScaleParams::kCellNumber10 + 7, 7, 1* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 8:call_onekernel_solve_fusedcompute(0* LstmScaleParams::kCellNumber10 + 8, 8, 0* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}