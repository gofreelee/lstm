#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_106(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(7, 99);break;case 1:call_onekernel_compute_wi(8, 98);break;case 2:call_onekernel_compute_wi(9, 97);break;case 3:call_onekernel_compute_uh(7, 99);break;case 4:call_onekernel_compute_uh(8, 98);break;case 5:call_onekernel_compute_uh(9, 97);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_106(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(7, 99);break;case 1:call_onekernel_solve(8, 98);break;case 2:call_onekernel_solve(9, 97);break;}
}