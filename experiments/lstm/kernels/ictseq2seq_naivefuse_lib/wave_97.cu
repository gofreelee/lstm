#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave_compute_97(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 2) {
case 0:call_onekernel_compute_naivefuse(97* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0, 97* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 1:call_onekernel_compute_naivefuse(96* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1, 96* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 2:call_onekernel_compute_naivefuse(95* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2, 95* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 3:call_onekernel_compute_naivefuse(94* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3, 94* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 4:call_onekernel_compute_naivefuse(93* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4, 93* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 5:call_onekernel_compute_naivefuse(92* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5, 92* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 6:call_onekernel_compute_naivefuse(91* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6, 91* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 7:call_onekernel_compute_naivefuse(90* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, 7, 90* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 8:call_onekernel_compute_naivefuse(89* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, 8, 89* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 9:call_onekernel_compute_naivefuse(88* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, 9, 88* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_97(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 2) {
case 0:call_onekernel_solve_naivefuse(97* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0, 97* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 1:call_onekernel_solve_naivefuse(96* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1, 96* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 2:call_onekernel_solve_naivefuse(95* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2, 95* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 3:call_onekernel_solve_naivefuse(94* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3, 94* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 4:call_onekernel_solve_naivefuse(93* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4, 93* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 5:call_onekernel_solve_naivefuse(92* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5, 92* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 6:call_onekernel_solve_naivefuse(91* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6, 91* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 7:call_onekernel_solve_naivefuse(90* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, 7, 90* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 7, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 8:call_onekernel_solve_naivefuse(89* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, 8, 89* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 8, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;case 9:call_onekernel_solve_naivefuse(88* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, 9, 88* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 9, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);break;}
}