#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave95(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 4) {
case 0:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(95* LstmScaleParams::kCellNumber10 + 0, 0, 95* LstmScaleParams::kCellNumber10 + 0, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 1:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(94* LstmScaleParams::kCellNumber10 + 1, 1, 94* LstmScaleParams::kCellNumber10 + 1, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 2:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(93* LstmScaleParams::kCellNumber10 + 2, 2, 93* LstmScaleParams::kCellNumber10 + 2, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 3:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(92* LstmScaleParams::kCellNumber10 + 3, 3, 92* LstmScaleParams::kCellNumber10 + 3, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 4:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(91* LstmScaleParams::kCellNumber10 + 4, 4, 91* LstmScaleParams::kCellNumber10 + 4, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 5:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(90* LstmScaleParams::kCellNumber10 + 5, 5, 90* LstmScaleParams::kCellNumber10 + 5, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 6:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(89* LstmScaleParams::kCellNumber10 + 6, 6, 89* LstmScaleParams::kCellNumber10 + 6, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 7:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(88* LstmScaleParams::kCellNumber10 + 7, 7, 88* LstmScaleParams::kCellNumber10 + 7, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 8:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(87* LstmScaleParams::kCellNumber10 + 8, 8, 87* LstmScaleParams::kCellNumber10 + 8, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;case 9:call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_16blocks_eachcell(86* LstmScaleParams::kCellNumber10 + 9, 9, 86* LstmScaleParams::kCellNumber10 + 9, 16, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, 15);break;}
}