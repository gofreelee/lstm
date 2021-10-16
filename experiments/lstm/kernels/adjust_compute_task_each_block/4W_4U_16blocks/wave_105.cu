#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_105(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(6, 99);break;case 1:call_onekernel_compute_wi(7, 98);break;case 2:call_onekernel_compute_wi(8, 97);break;case 3:call_onekernel_compute_wi(9, 96);break;case 4:call_onekernel_compute_uh(6, 99);break;case 5:call_onekernel_compute_uh(7, 98);break;case 6:call_onekernel_compute_uh(8, 97);break;case 7:call_onekernel_compute_uh(9, 96);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_105(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(6, 99);break;case 1:call_onekernel_solve(7, 98);break;case 2:call_onekernel_solve(8, 97);break;case 3:call_onekernel_solve(9, 96);break;}
}