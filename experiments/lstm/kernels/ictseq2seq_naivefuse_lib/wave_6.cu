#include "kernels/include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave_compute_6(WaveInputParams *__restrict__ input,
                   WaveModelParams *__restrict__ model,
                   WaveOutputParams *__restrict__ output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("base inputaddr :%d\n", input + 48);
        printf("base inputaddr :%d\n", input);
    }
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_compute_naivefuse(
            6 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0,
            6 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 1:
        call_onekernel_compute_naivefuse(
            5 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1,
            5 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);


        break;
    case 2:

        call_onekernel_compute_naivefuse(
            4 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2,
            4 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 3:

        call_onekernel_compute_naivefuse(
            3 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3,
            3 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);

        break;
    case 4:

        call_onekernel_compute_naivefuse(
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4,
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);

        break;
    case 5:

        call_onekernel_compute_naivefuse(
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5,
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);

        break;
    case 6:

        call_onekernel_compute_naivefuse(
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6,
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);

        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave_solve_6(WaveInputParams *__restrict__ input,
                 WaveModelParams *__restrict__ model,
                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 2) {
    case 0:
        call_onekernel_solve_naivefuse(
            6 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0, 0,
            6 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 1:
        call_onekernel_solve_naivefuse(
            5 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1, 1,
            5 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 2:
        call_onekernel_solve_naivefuse(
            4 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2, 2,
            4 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 3:
        call_onekernel_solve_naivefuse(
            3 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3, 3,
            3 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 3,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 4:
        call_onekernel_solve_naivefuse(
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4, 4,
            2 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 4,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 5:
        call_onekernel_solve_naivefuse(
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5, 5,
            1 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 5,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    case 6:
        call_onekernel_solve_naivefuse(
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6, 6,
            0 * LstmScaleParams::kSeq2SeqEncodeCellNumber8 + 6,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128,
            LstmScaleParams::kInputSize128,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);
        break;
    }
}