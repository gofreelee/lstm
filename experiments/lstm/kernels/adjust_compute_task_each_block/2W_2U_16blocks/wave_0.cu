#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)
    wave_compute_0(WaveInputParams *__restrict__ input,
                   WaveModelParams *__restrict__ model,
                   WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_compute_2_wi_uh_0(0, 0);
        break;
    case 1:
        call_onekernel_compute_2_wi_uh_1(0, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave_solve_0(WaveInputParams *__restrict__ input,
                 WaveModelParams *__restrict__ model,
                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_solve(0, 0);
        break;
    }
}