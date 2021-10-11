#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_108(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(9, 99);break;case 1:call_onekernel_compute_uh(9, 99);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_108(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(9, 99);break;}
}