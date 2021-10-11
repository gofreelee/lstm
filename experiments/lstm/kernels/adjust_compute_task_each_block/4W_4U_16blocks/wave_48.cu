#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_48(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 48);break;case 1:call_onekernel_compute_wi(1, 47);break;case 2:call_onekernel_compute_wi(2, 46);break;case 3:call_onekernel_compute_wi(3, 45);break;case 4:call_onekernel_compute_wi(4, 44);break;case 5:call_onekernel_compute_wi(5, 43);break;case 6:call_onekernel_compute_wi(6, 42);break;case 7:call_onekernel_compute_wi(7, 41);break;case 8:call_onekernel_compute_wi(8, 40);break;case 9:call_onekernel_compute_wi(9, 39);break;case 10:call_onekernel_compute_uh(0, 48);break;case 11:call_onekernel_compute_uh(1, 47);break;case 12:call_onekernel_compute_uh(2, 46);break;case 13:call_onekernel_compute_uh(3, 45);break;case 14:call_onekernel_compute_uh(4, 44);break;case 15:call_onekernel_compute_uh(5, 43);break;case 16:call_onekernel_compute_uh(6, 42);break;case 17:call_onekernel_compute_uh(7, 41);break;case 18:call_onekernel_compute_uh(8, 40);break;case 19:call_onekernel_compute_uh(9, 39);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_48(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 48);break;case 1:call_onekernel_solve(1, 47);break;case 2:call_onekernel_solve(2, 46);break;case 3:call_onekernel_solve(3, 45);break;case 4:call_onekernel_solve(4, 44);break;case 5:call_onekernel_solve(5, 43);break;case 6:call_onekernel_solve(6, 42);break;case 7:call_onekernel_solve(7, 41);break;case 8:call_onekernel_solve(8, 40);break;case 9:call_onekernel_solve(9, 39);break;}
}