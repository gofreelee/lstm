#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave26(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 4) {
case 0:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(26* LstmScaleParams::kCellNumber10 + 0, 0, 26* LstmScaleParams::kCellNumber10 + 0, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 1:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(25* LstmScaleParams::kCellNumber10 + 1, 1, 25* LstmScaleParams::kCellNumber10 + 1, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 2:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(24* LstmScaleParams::kCellNumber10 + 2, 2, 24* LstmScaleParams::kCellNumber10 + 2, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 3:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(23* LstmScaleParams::kCellNumber10 + 3, 3, 23* LstmScaleParams::kCellNumber10 + 3, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 4:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(22* LstmScaleParams::kCellNumber10 + 4, 4, 22* LstmScaleParams::kCellNumber10 + 4, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 5:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(21* LstmScaleParams::kCellNumber10 + 5, 5, 21* LstmScaleParams::kCellNumber10 + 5, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 6:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(20* LstmScaleParams::kCellNumber10 + 6, 6, 20* LstmScaleParams::kCellNumber10 + 6, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 7:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(19* LstmScaleParams::kCellNumber10 + 7, 7, 19* LstmScaleParams::kCellNumber10 + 7, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 8:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(18* LstmScaleParams::kCellNumber10 + 8, 8, 18* LstmScaleParams::kCellNumber10 + 8, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 9:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell_float4(17* LstmScaleParams::kCellNumber10 + 9, 9, 17* LstmScaleParams::kCellNumber10 + 9, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;}
}