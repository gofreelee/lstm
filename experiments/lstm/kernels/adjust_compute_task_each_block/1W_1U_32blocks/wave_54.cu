#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_54(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 54);break;case 1:call_onekernel_compute_wi_uh_0(1, 53);break;case 2:call_onekernel_compute_wi_uh_0(2, 52);break;case 3:call_onekernel_compute_wi_uh_0(3, 51);break;case 4:call_onekernel_compute_wi_uh_0(4, 50);break;case 5:call_onekernel_compute_wi_uh_0(5, 49);break;case 6:call_onekernel_compute_wi_uh_0(6, 48);break;case 7:call_onekernel_compute_wi_uh_0(7, 47);break;case 8:call_onekernel_compute_wi_uh_0(8, 46);break;case 9:call_onekernel_compute_wi_uh_0(9, 45);break;case 10:call_onekernel_compute_wi_uh_1(0, 54);break;case 11:call_onekernel_compute_wi_uh_1(1, 53);break;case 12:call_onekernel_compute_wi_uh_1(2, 52);break;case 13:call_onekernel_compute_wi_uh_1(3, 51);break;case 14:call_onekernel_compute_wi_uh_1(4, 50);break;case 15:call_onekernel_compute_wi_uh_1(5, 49);break;case 16:call_onekernel_compute_wi_uh_1(6, 48);break;case 17:call_onekernel_compute_wi_uh_1(7, 47);break;case 18:call_onekernel_compute_wi_uh_1(8, 46);break;case 19:call_onekernel_compute_wi_uh_1(9, 45);break;case 20:call_onekernel_compute_wi_uh_2(0, 54);break;case 21:call_onekernel_compute_wi_uh_2(1, 53);break;case 22:call_onekernel_compute_wi_uh_2(2, 52);break;case 23:call_onekernel_compute_wi_uh_2(3, 51);break;case 24:call_onekernel_compute_wi_uh_2(4, 50);break;case 25:call_onekernel_compute_wi_uh_2(5, 49);break;case 26:call_onekernel_compute_wi_uh_2(6, 48);break;case 27:call_onekernel_compute_wi_uh_2(7, 47);break;case 28:call_onekernel_compute_wi_uh_2(8, 46);break;case 29:call_onekernel_compute_wi_uh_2(9, 45);break;case 30:call_onekernel_compute_wi_uh_3(0, 54);break;case 31:call_onekernel_compute_wi_uh_3(1, 53);break;case 32:call_onekernel_compute_wi_uh_3(2, 52);break;case 33:call_onekernel_compute_wi_uh_3(3, 51);break;case 34:call_onekernel_compute_wi_uh_3(4, 50);break;case 35:call_onekernel_compute_wi_uh_3(5, 49);break;case 36:call_onekernel_compute_wi_uh_3(6, 48);break;case 37:call_onekernel_compute_wi_uh_3(7, 47);break;case 38:call_onekernel_compute_wi_uh_3(8, 46);break;case 39:call_onekernel_compute_wi_uh_3(9, 45);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_54(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 54);break;case 1:call_onekernel_solve(1, 53);break;case 2:call_onekernel_solve(2, 52);break;case 3:call_onekernel_solve(3, 51);break;case 4:call_onekernel_solve(4, 50);break;case 5:call_onekernel_solve(5, 49);break;case 6:call_onekernel_solve(6, 48);break;case 7:call_onekernel_solve(7, 47);break;case 8:call_onekernel_solve(8, 46);break;case 9:call_onekernel_solve(9, 45);break;}
}