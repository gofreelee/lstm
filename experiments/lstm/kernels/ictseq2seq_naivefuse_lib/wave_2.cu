#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave_compute_2(WaveInputParams *__restrict__ input,
                   WaveModelParams *__restrict__ model,
                   WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_naivefuse(
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0,
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 1:
        call_onekernel_compute_naivefuse(
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1,
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 2:
        call_onekernel_compute_naivefuse(
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2,
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave_solve_2(WaveInputParams *__restrict__ input,
                 WaveModelParams *__restrict__ model,
                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_naivefuse(
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0,
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 1:
        call_onekernel_solve_naivefuse(
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1,
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 2:
        call_onekernel_solve_naivefuse(
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2,
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    }
}