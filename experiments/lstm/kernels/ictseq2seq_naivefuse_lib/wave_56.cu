#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave_compute_56(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 2) {
case 0:call_onekernel_compute_naivefuse(56* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0, 56* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 1:call_onekernel_compute_naivefuse(55* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1, 55* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 2:call_onekernel_compute_naivefuse(54* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2, 54* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 3:call_onekernel_compute_naivefuse(53* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3, 53* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 4:call_onekernel_compute_naivefuse(52* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4, 52* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 5:call_onekernel_compute_naivefuse(51* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5, 51* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 6:call_onekernel_compute_naivefuse(50* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6, 50* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 7:call_onekernel_compute_naivefuse(49* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, 7, 49* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 8:call_onekernel_compute_naivefuse(48* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, 8, 48* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 9:call_onekernel_compute_naivefuse(47* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, 9, 47* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_56(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 2) {
case 0:call_onekernel_solve_naivefuse(56* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0, 56* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 1:call_onekernel_solve_naivefuse(55* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1, 55* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 2:call_onekernel_solve_naivefuse(54* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2, 54* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 3:call_onekernel_solve_naivefuse(53* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3, 53* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 4:call_onekernel_solve_naivefuse(52* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4, 52* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 5:call_onekernel_solve_naivefuse(51* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5, 51* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 6:call_onekernel_solve_naivefuse(50* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6, 50* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 7:call_onekernel_solve_naivefuse(49* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, 7, 49* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 8:call_onekernel_solve_naivefuse(48* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, 8, 48* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 9:call_onekernel_solve_naivefuse(47* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, 9, 47* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;}
}