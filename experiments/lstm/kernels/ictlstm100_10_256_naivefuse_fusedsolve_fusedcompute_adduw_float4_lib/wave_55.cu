#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave55(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(55* LstmScaleParams::kCellNumber10 + 0, 0, 55* LstmScaleParams::kCellNumber10 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 1:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(54* LstmScaleParams::kCellNumber10 + 1, 1, 54* LstmScaleParams::kCellNumber10 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 2:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(53* LstmScaleParams::kCellNumber10 + 2, 2, 53* LstmScaleParams::kCellNumber10 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 3:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(52* LstmScaleParams::kCellNumber10 + 3, 3, 52* LstmScaleParams::kCellNumber10 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 4:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(51* LstmScaleParams::kCellNumber10 + 4, 4, 51* LstmScaleParams::kCellNumber10 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 5:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(50* LstmScaleParams::kCellNumber10 + 5, 5, 50* LstmScaleParams::kCellNumber10 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 6:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(49* LstmScaleParams::kCellNumber10 + 6, 6, 49* LstmScaleParams::kCellNumber10 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 7:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(48* LstmScaleParams::kCellNumber10 + 7, 7, 48* LstmScaleParams::kCellNumber10 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 8:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(47* LstmScaleParams::kCellNumber10 + 8, 8, 47* LstmScaleParams::kCellNumber10 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;case 9:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(46* LstmScaleParams::kCellNumber10 + 9, 9, 46* LstmScaleParams::kCellNumber10 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);break;}
}