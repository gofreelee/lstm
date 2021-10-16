#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_1(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 1);break;case 1:call_onekernel_compute_wi(1, 0);break;case 2:call_onekernel_compute_uh(0, 1);break;case 3:call_onekernel_compute_uh(1, 0);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_1(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 1);break;case 1:call_onekernel_solve(1, 0);break;}
}