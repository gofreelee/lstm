#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_7(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 7);break;case 1:call_onekernel_compute_wi(1, 6);break;case 2:call_onekernel_compute_wi(2, 5);break;case 3:call_onekernel_compute_wi(3, 4);break;case 4:call_onekernel_compute_wi(4, 3);break;case 5:call_onekernel_compute_wi(5, 2);break;case 6:call_onekernel_compute_wi(6, 1);break;case 7:call_onekernel_compute_wi(7, 0);break;case 8:call_onekernel_compute_uh(0, 7);break;case 9:call_onekernel_compute_uh(1, 6);break;case 10:call_onekernel_compute_uh(2, 5);break;case 11:call_onekernel_compute_uh(3, 4);break;case 12:call_onekernel_compute_uh(4, 3);break;case 13:call_onekernel_compute_uh(5, 2);break;case 14:call_onekernel_compute_uh(6, 1);break;case 15:call_onekernel_compute_uh(7, 0);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_7(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 7);break;case 1:call_onekernel_solve(1, 6);break;case 2:call_onekernel_solve(2, 5);break;case 3:call_onekernel_solve(3, 4);break;case 4:call_onekernel_solve(4, 3);break;case 5:call_onekernel_solve(5, 2);break;case 6:call_onekernel_solve(6, 1);break;case 7:call_onekernel_solve(7, 0);break;}
}