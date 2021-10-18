#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_4(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 4);break;case 1:call_onekernel_compute_wi_uh_0(1, 3);break;case 2:call_onekernel_compute_wi_uh_0(2, 2);break;case 3:call_onekernel_compute_wi_uh_0(3, 1);break;case 4:call_onekernel_compute_wi_uh_0(4, 0);break;case 5:call_onekernel_compute_wi_uh_1(0, 4);break;case 6:call_onekernel_compute_wi_uh_1(1, 3);break;case 7:call_onekernel_compute_wi_uh_1(2, 2);break;case 8:call_onekernel_compute_wi_uh_1(3, 1);break;case 9:call_onekernel_compute_wi_uh_1(4, 0);break;case 10:call_onekernel_compute_wi_uh_2(0, 4);break;case 11:call_onekernel_compute_wi_uh_2(1, 3);break;case 12:call_onekernel_compute_wi_uh_2(2, 2);break;case 13:call_onekernel_compute_wi_uh_2(3, 1);break;case 14:call_onekernel_compute_wi_uh_2(4, 0);break;case 15:call_onekernel_compute_wi_uh_3(0, 4);break;case 16:call_onekernel_compute_wi_uh_3(1, 3);break;case 17:call_onekernel_compute_wi_uh_3(2, 2);break;case 18:call_onekernel_compute_wi_uh_3(3, 1);break;case 19:call_onekernel_compute_wi_uh_3(4, 0);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_4(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 4);break;case 1:call_onekernel_solve(1, 3);break;case 2:call_onekernel_solve(2, 2);break;case 3:call_onekernel_solve(3, 1);break;case 4:call_onekernel_solve(4, 0);break;}
}