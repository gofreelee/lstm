#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_116(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_0(17, 99);break;case 1:call_onekernel_compute_wi_0(18, 98);break;case 2:call_onekernel_compute_wi_0(19, 97);break;case 3:call_onekernel_compute_wi_1(17, 99);break;case 4:call_onekernel_compute_wi_1(18, 98);break;case 5:call_onekernel_compute_wi_1(19, 97);break;case 6:call_onekernel_compute_wi_2(17, 99);break;case 7:call_onekernel_compute_wi_2(18, 98);break;case 8:call_onekernel_compute_wi_2(19, 97);break;case 9:call_onekernel_compute_wi_3(17, 99);break;case 10:call_onekernel_compute_wi_3(18, 98);break;case 11:call_onekernel_compute_wi_3(19, 97);break;case 12:call_onekernel_compute_uh_0(17, 99);break;case 13:call_onekernel_compute_uh_0(18, 98);break;case 14:call_onekernel_compute_uh_0(19, 97);break;case 15:call_onekernel_compute_uh_1(17, 99);break;case 16:call_onekernel_compute_uh_1(18, 98);break;case 17:call_onekernel_compute_uh_1(19, 97);break;case 18:call_onekernel_compute_uh_2(17, 99);break;case 19:call_onekernel_compute_uh_2(18, 98);break;case 20:call_onekernel_compute_uh_2(19, 97);break;case 21:call_onekernel_compute_uh_3(17, 99);break;case 22:call_onekernel_compute_uh_3(18, 98);break;case 23:call_onekernel_compute_uh_3(19, 97);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_116(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(17, 99);break;case 1:call_onekernel_solve(18, 98);break;case 2:call_onekernel_solve(19, 97);break;}
}