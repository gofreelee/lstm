#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_51(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_2_wi_uh_0(0, 51);break;case 1:call_onekernel_compute_2_wi_uh_0(1, 50);break;case 2:call_onekernel_compute_2_wi_uh_0(2, 49);break;case 3:call_onekernel_compute_2_wi_uh_0(3, 48);break;case 4:call_onekernel_compute_2_wi_uh_0(4, 47);break;case 5:call_onekernel_compute_2_wi_uh_0(5, 46);break;case 6:call_onekernel_compute_2_wi_uh_0(6, 45);break;case 7:call_onekernel_compute_2_wi_uh_0(7, 44);break;case 8:call_onekernel_compute_2_wi_uh_0(8, 43);break;case 9:call_onekernel_compute_2_wi_uh_0(9, 42);break;case 10:call_onekernel_compute_2_wi_uh_1(0, 51);break;case 11:call_onekernel_compute_2_wi_uh_1(1, 50);break;case 12:call_onekernel_compute_2_wi_uh_1(2, 49);break;case 13:call_onekernel_compute_2_wi_uh_1(3, 48);break;case 14:call_onekernel_compute_2_wi_uh_1(4, 47);break;case 15:call_onekernel_compute_2_wi_uh_1(5, 46);break;case 16:call_onekernel_compute_2_wi_uh_1(6, 45);break;case 17:call_onekernel_compute_2_wi_uh_1(7, 44);break;case 18:call_onekernel_compute_2_wi_uh_1(8, 43);break;case 19:call_onekernel_compute_2_wi_uh_1(9, 42);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_51(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 51);break;case 1:call_onekernel_solve(1, 50);break;case 2:call_onekernel_solve(2, 49);break;case 3:call_onekernel_solve(3, 48);break;case 4:call_onekernel_solve(4, 47);break;case 5:call_onekernel_solve(5, 46);break;case 6:call_onekernel_solve(6, 45);break;case 7:call_onekernel_solve(7, 44);break;case 8:call_onekernel_solve(8, 43);break;case 9:call_onekernel_solve(9, 42);break;}
}