#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)
    wave_compute_2(WaveInputParams *__restrict__ input,
                   WaveModelParams *__restrict__ model,
                   WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_compute_wi_uh_0(0, 2);
        break;
    case 1:
        call_onekernel_compute_wi_uh_0(1, 1);
        break;
    case 2:
        call_onekernel_compute_wi_uh_0(2, 0);
        break;
    case 3:
        call_onekernel_compute_wi_uh_1(0, 2);
        break;
    case 4:
        call_onekernel_compute_wi_uh_1(1, 1);
        break;
    case 5:
        call_onekernel_compute_wi_uh_1(2, 0);
        break;
    case 6:
        call_onekernel_compute_wi_uh_2(0, 2);
        break;
    case 7:
        call_onekernel_compute_wi_uh_2(1, 1);
        break;
    case 8:
        call_onekernel_compute_wi_uh_2(2, 0);
        break;
    case 9:
        call_onekernel_compute_wi_uh_3(0, 2);
        break;
    case 10:
        call_onekernel_compute_wi_uh_3(1, 1);
        break;
    case 11:
        call_onekernel_compute_wi_uh_3(2, 0);
        break;
    }
}
__global__ void __launch_bounds__(256, 1)
    wave_solve_2(WaveInputParams *__restrict__ input,
                 WaveModelParams *__restrict__ model,
                 WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_solve(0, 2);
        break;
    case 1:
        call_onekernel_solve(1, 1);
        break;
    case 2:
        call_onekernel_solve(2, 0);
        break;
    }
}