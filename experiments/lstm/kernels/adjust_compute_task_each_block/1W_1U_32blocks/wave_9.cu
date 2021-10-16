#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_9(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 9);break;case 1:call_onekernel_compute_wi_uh_0(1, 8);break;case 2:call_onekernel_compute_wi_uh_0(2, 7);break;case 3:call_onekernel_compute_wi_uh_0(3, 6);break;case 4:call_onekernel_compute_wi_uh_0(4, 5);break;case 5:call_onekernel_compute_wi_uh_0(5, 4);break;case 6:call_onekernel_compute_wi_uh_0(6, 3);break;case 7:call_onekernel_compute_wi_uh_0(7, 2);break;case 8:call_onekernel_compute_wi_uh_0(8, 1);break;case 9:call_onekernel_compute_wi_uh_0(9, 0);break;case 10:call_onekernel_compute_wi_uh_1(0, 9);break;case 11:call_onekernel_compute_wi_uh_1(1, 8);break;case 12:call_onekernel_compute_wi_uh_1(2, 7);break;case 13:call_onekernel_compute_wi_uh_1(3, 6);break;case 14:call_onekernel_compute_wi_uh_1(4, 5);break;case 15:call_onekernel_compute_wi_uh_1(5, 4);break;case 16:call_onekernel_compute_wi_uh_1(6, 3);break;case 17:call_onekernel_compute_wi_uh_1(7, 2);break;case 18:call_onekernel_compute_wi_uh_1(8, 1);break;case 19:call_onekernel_compute_wi_uh_1(9, 0);break;case 20:call_onekernel_compute_wi_uh_2(0, 9);break;case 21:call_onekernel_compute_wi_uh_2(1, 8);break;case 22:call_onekernel_compute_wi_uh_2(2, 7);break;case 23:call_onekernel_compute_wi_uh_2(3, 6);break;case 24:call_onekernel_compute_wi_uh_2(4, 5);break;case 25:call_onekernel_compute_wi_uh_2(5, 4);break;case 26:call_onekernel_compute_wi_uh_2(6, 3);break;case 27:call_onekernel_compute_wi_uh_2(7, 2);break;case 28:call_onekernel_compute_wi_uh_2(8, 1);break;case 29:call_onekernel_compute_wi_uh_2(9, 0);break;case 30:call_onekernel_compute_wi_uh_3(0, 9);break;case 31:call_onekernel_compute_wi_uh_3(1, 8);break;case 32:call_onekernel_compute_wi_uh_3(2, 7);break;case 33:call_onekernel_compute_wi_uh_3(3, 6);break;case 34:call_onekernel_compute_wi_uh_3(4, 5);break;case 35:call_onekernel_compute_wi_uh_3(5, 4);break;case 36:call_onekernel_compute_wi_uh_3(6, 3);break;case 37:call_onekernel_compute_wi_uh_3(7, 2);break;case 38:call_onekernel_compute_wi_uh_3(8, 1);break;case 39:call_onekernel_compute_wi_uh_3(9, 0);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_9(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 9);break;case 1:call_onekernel_solve(1, 8);break;case 2:call_onekernel_solve(2, 7);break;case 3:call_onekernel_solve(3, 6);break;case 4:call_onekernel_solve(4, 5);break;case 5:call_onekernel_solve(5, 4);break;case 6:call_onekernel_solve(6, 3);break;case 7:call_onekernel_solve(7, 2);break;case 8:call_onekernel_solve(8, 1);break;case 9:call_onekernel_solve(9, 0);break;}
}