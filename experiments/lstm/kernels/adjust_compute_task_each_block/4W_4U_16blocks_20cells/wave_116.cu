#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_116(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(17, 99);break;case 1:call_onekernel_compute_wi(18, 98);break;case 2:call_onekernel_compute_wi(19, 97);break;case 3:call_onekernel_compute_uh(17, 99);break;case 4:call_onekernel_compute_uh(18, 98);break;case 5:call_onekernel_compute_uh(19, 97);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_116(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(17, 99);break;case 1:call_onekernel_solve(18, 98);break;case 2:call_onekernel_solve(19, 97);break;}
}