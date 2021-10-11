#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_5(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 5);break;case 1:call_onekernel_compute_wi_uh_0(1, 4);break;case 2:call_onekernel_compute_wi_uh_0(2, 3);break;case 3:call_onekernel_compute_wi_uh_0(3, 2);break;case 4:call_onekernel_compute_wi_uh_0(4, 1);break;case 5:call_onekernel_compute_wi_uh_0(5, 0);break;case 6:call_onekernel_compute_wi_uh_1(0, 5);break;case 7:call_onekernel_compute_wi_uh_1(1, 4);break;case 8:call_onekernel_compute_wi_uh_1(2, 3);break;case 9:call_onekernel_compute_wi_uh_1(3, 2);break;case 10:call_onekernel_compute_wi_uh_1(4, 1);break;case 11:call_onekernel_compute_wi_uh_1(5, 0);break;case 12:call_onekernel_compute_wi_uh_2(0, 5);break;case 13:call_onekernel_compute_wi_uh_2(1, 4);break;case 14:call_onekernel_compute_wi_uh_2(2, 3);break;case 15:call_onekernel_compute_wi_uh_2(3, 2);break;case 16:call_onekernel_compute_wi_uh_2(4, 1);break;case 17:call_onekernel_compute_wi_uh_2(5, 0);break;case 18:call_onekernel_compute_wi_uh_3(0, 5);break;case 19:call_onekernel_compute_wi_uh_3(1, 4);break;case 20:call_onekernel_compute_wi_uh_3(2, 3);break;case 21:call_onekernel_compute_wi_uh_3(3, 2);break;case 22:call_onekernel_compute_wi_uh_3(4, 1);break;case 23:call_onekernel_compute_wi_uh_3(5, 0);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_5(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 5);break;case 1:call_onekernel_solve(1, 4);break;case 2:call_onekernel_solve(2, 3);break;case 3:call_onekernel_solve(3, 2);break;case 4:call_onekernel_solve(4, 1);break;case 5:call_onekernel_solve(5, 0);break;}
}