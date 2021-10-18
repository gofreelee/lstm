#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_12(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 12);break;case 1:call_onekernel_compute_wi(1, 11);break;case 2:call_onekernel_compute_wi(2, 10);break;case 3:call_onekernel_compute_wi(3, 9);break;case 4:call_onekernel_compute_wi(4, 8);break;case 5:call_onekernel_compute_wi(5, 7);break;case 6:call_onekernel_compute_wi(6, 6);break;case 7:call_onekernel_compute_wi(7, 5);break;case 8:call_onekernel_compute_wi(8, 4);break;case 9:call_onekernel_compute_wi(9, 3);break;case 10:call_onekernel_compute_wi(10, 2);break;case 11:call_onekernel_compute_wi(11, 1);break;case 12:call_onekernel_compute_wi(12, 0);break;case 13:call_onekernel_compute_uh(0, 12);break;case 14:call_onekernel_compute_uh(1, 11);break;case 15:call_onekernel_compute_uh(2, 10);break;case 16:call_onekernel_compute_uh(3, 9);break;case 17:call_onekernel_compute_uh(4, 8);break;case 18:call_onekernel_compute_uh(5, 7);break;case 19:call_onekernel_compute_uh(6, 6);break;case 20:call_onekernel_compute_uh(7, 5);break;case 21:call_onekernel_compute_uh(8, 4);break;case 22:call_onekernel_compute_uh(9, 3);break;case 23:call_onekernel_compute_uh(10, 2);break;case 24:call_onekernel_compute_uh(11, 1);break;case 25:call_onekernel_compute_uh(12, 0);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_12(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 12);break;case 1:call_onekernel_solve(1, 11);break;case 2:call_onekernel_solve(2, 10);break;case 3:call_onekernel_solve(3, 9);break;case 4:call_onekernel_solve(4, 8);break;case 5:call_onekernel_solve(5, 7);break;case 6:call_onekernel_solve(6, 6);break;case 7:call_onekernel_solve(7, 5);break;case 8:call_onekernel_solve(8, 4);break;case 9:call_onekernel_solve(9, 3);break;case 10:call_onekernel_solve(10, 2);break;case 11:call_onekernel_solve(11, 1);break;case 12:call_onekernel_solve(12, 0);break;}
}