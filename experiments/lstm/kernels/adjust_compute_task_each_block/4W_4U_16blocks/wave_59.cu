#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_59(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 59);break;case 1:call_onekernel_compute_wi(1, 58);break;case 2:call_onekernel_compute_wi(2, 57);break;case 3:call_onekernel_compute_wi(3, 56);break;case 4:call_onekernel_compute_wi(4, 55);break;case 5:call_onekernel_compute_wi(5, 54);break;case 6:call_onekernel_compute_wi(6, 53);break;case 7:call_onekernel_compute_wi(7, 52);break;case 8:call_onekernel_compute_wi(8, 51);break;case 9:call_onekernel_compute_wi(9, 50);break;case 10:call_onekernel_compute_uh(0, 59);break;case 11:call_onekernel_compute_uh(1, 58);break;case 12:call_onekernel_compute_uh(2, 57);break;case 13:call_onekernel_compute_uh(3, 56);break;case 14:call_onekernel_compute_uh(4, 55);break;case 15:call_onekernel_compute_uh(5, 54);break;case 16:call_onekernel_compute_uh(6, 53);break;case 17:call_onekernel_compute_uh(7, 52);break;case 18:call_onekernel_compute_uh(8, 51);break;case 19:call_onekernel_compute_uh(9, 50);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_59(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 59);break;case 1:call_onekernel_solve(1, 58);break;case 2:call_onekernel_solve(2, 57);break;case 3:call_onekernel_solve(3, 56);break;case 4:call_onekernel_solve(4, 55);break;case 5:call_onekernel_solve(5, 54);break;case 6:call_onekernel_solve(6, 53);break;case 7:call_onekernel_solve(7, 52);break;case 8:call_onekernel_solve(8, 51);break;case 9:call_onekernel_solve(9, 50);break;}
}