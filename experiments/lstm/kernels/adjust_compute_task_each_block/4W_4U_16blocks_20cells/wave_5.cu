#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_5(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 5);break;case 1:call_onekernel_compute_wi(1, 4);break;case 2:call_onekernel_compute_wi(2, 3);break;case 3:call_onekernel_compute_wi(3, 2);break;case 4:call_onekernel_compute_wi(4, 1);break;case 5:call_onekernel_compute_wi(5, 0);break;case 6:call_onekernel_compute_uh(0, 5);break;case 7:call_onekernel_compute_uh(1, 4);break;case 8:call_onekernel_compute_uh(2, 3);break;case 9:call_onekernel_compute_uh(3, 2);break;case 10:call_onekernel_compute_uh(4, 1);break;case 11:call_onekernel_compute_uh(5, 0);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_5(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 5);break;case 1:call_onekernel_solve(1, 4);break;case 2:call_onekernel_solve(2, 3);break;case 3:call_onekernel_solve(3, 2);break;case 4:call_onekernel_solve(4, 1);break;case 5:call_onekernel_solve(5, 0);break;}
}