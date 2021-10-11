#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_57(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 57);break;case 1:call_onekernel_compute_wi_uh_0(1, 56);break;case 2:call_onekernel_compute_wi_uh_0(2, 55);break;case 3:call_onekernel_compute_wi_uh_0(3, 54);break;case 4:call_onekernel_compute_wi_uh_0(4, 53);break;case 5:call_onekernel_compute_wi_uh_0(5, 52);break;case 6:call_onekernel_compute_wi_uh_0(6, 51);break;case 7:call_onekernel_compute_wi_uh_0(7, 50);break;case 8:call_onekernel_compute_wi_uh_0(8, 49);break;case 9:call_onekernel_compute_wi_uh_0(9, 48);break;case 10:call_onekernel_compute_wi_uh_1(0, 57);break;case 11:call_onekernel_compute_wi_uh_1(1, 56);break;case 12:call_onekernel_compute_wi_uh_1(2, 55);break;case 13:call_onekernel_compute_wi_uh_1(3, 54);break;case 14:call_onekernel_compute_wi_uh_1(4, 53);break;case 15:call_onekernel_compute_wi_uh_1(5, 52);break;case 16:call_onekernel_compute_wi_uh_1(6, 51);break;case 17:call_onekernel_compute_wi_uh_1(7, 50);break;case 18:call_onekernel_compute_wi_uh_1(8, 49);break;case 19:call_onekernel_compute_wi_uh_1(9, 48);break;case 20:call_onekernel_compute_wi_uh_2(0, 57);break;case 21:call_onekernel_compute_wi_uh_2(1, 56);break;case 22:call_onekernel_compute_wi_uh_2(2, 55);break;case 23:call_onekernel_compute_wi_uh_2(3, 54);break;case 24:call_onekernel_compute_wi_uh_2(4, 53);break;case 25:call_onekernel_compute_wi_uh_2(5, 52);break;case 26:call_onekernel_compute_wi_uh_2(6, 51);break;case 27:call_onekernel_compute_wi_uh_2(7, 50);break;case 28:call_onekernel_compute_wi_uh_2(8, 49);break;case 29:call_onekernel_compute_wi_uh_2(9, 48);break;case 30:call_onekernel_compute_wi_uh_3(0, 57);break;case 31:call_onekernel_compute_wi_uh_3(1, 56);break;case 32:call_onekernel_compute_wi_uh_3(2, 55);break;case 33:call_onekernel_compute_wi_uh_3(3, 54);break;case 34:call_onekernel_compute_wi_uh_3(4, 53);break;case 35:call_onekernel_compute_wi_uh_3(5, 52);break;case 36:call_onekernel_compute_wi_uh_3(6, 51);break;case 37:call_onekernel_compute_wi_uh_3(7, 50);break;case 38:call_onekernel_compute_wi_uh_3(8, 49);break;case 39:call_onekernel_compute_wi_uh_3(9, 48);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_57(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 57);break;case 1:call_onekernel_solve(1, 56);break;case 2:call_onekernel_solve(2, 55);break;case 3:call_onekernel_solve(3, 54);break;case 4:call_onekernel_solve(4, 53);break;case 5:call_onekernel_solve(5, 52);break;case 6:call_onekernel_solve(6, 51);break;case 7:call_onekernel_solve(7, 50);break;case 8:call_onekernel_solve(8, 49);break;case 9:call_onekernel_solve(9, 48);break;}
}