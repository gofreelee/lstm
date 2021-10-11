#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)wave_compute_172(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){call_onekernel_compute_naivefuse(16* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100, 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8, 16* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);}
__global__ void __launch_bounds__(256, 1)wave_solve_172(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){call_onekernel_solve_naivefuse(16* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100, 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8, 16* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + 1+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100, LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);}
