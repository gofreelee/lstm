#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave59(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 4) {
case 0:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(59* LstmScaleParams::kCellNumber10 + 0, 0, 59* LstmScaleParams::kCellNumber10 + 0, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 1:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(58* LstmScaleParams::kCellNumber10 + 1, 1, 58* LstmScaleParams::kCellNumber10 + 1, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 2:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(57* LstmScaleParams::kCellNumber10 + 2, 2, 57* LstmScaleParams::kCellNumber10 + 2, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 3:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(56* LstmScaleParams::kCellNumber10 + 3, 3, 56* LstmScaleParams::kCellNumber10 + 3, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 4:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(55* LstmScaleParams::kCellNumber10 + 4, 4, 55* LstmScaleParams::kCellNumber10 + 4, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 5:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(54* LstmScaleParams::kCellNumber10 + 5, 5, 54* LstmScaleParams::kCellNumber10 + 5, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 6:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(53* LstmScaleParams::kCellNumber10 + 6, 6, 53* LstmScaleParams::kCellNumber10 + 6, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 7:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(52* LstmScaleParams::kCellNumber10 + 7, 7, 52* LstmScaleParams::kCellNumber10 + 7, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 8:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(51* LstmScaleParams::kCellNumber10 + 8, 8, 51* LstmScaleParams::kCellNumber10 + 8, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 9:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(50* LstmScaleParams::kCellNumber10 + 9, 9, 50* LstmScaleParams::kCellNumber10 + 9, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;}
}