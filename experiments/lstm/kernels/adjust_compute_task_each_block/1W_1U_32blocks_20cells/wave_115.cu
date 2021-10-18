#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_115(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(16, 99);break;case 1:call_onekernel_compute_wi_uh_0(17, 98);break;case 2:call_onekernel_compute_wi_uh_0(18, 97);break;case 3:call_onekernel_compute_wi_uh_0(19, 96);break;case 4:call_onekernel_compute_wi_uh_1(16, 99);break;case 5:call_onekernel_compute_wi_uh_1(17, 98);break;case 6:call_onekernel_compute_wi_uh_1(18, 97);break;case 7:call_onekernel_compute_wi_uh_1(19, 96);break;case 8:call_onekernel_compute_wi_uh_2(16, 99);break;case 9:call_onekernel_compute_wi_uh_2(17, 98);break;case 10:call_onekernel_compute_wi_uh_2(18, 97);break;case 11:call_onekernel_compute_wi_uh_2(19, 96);break;case 12:call_onekernel_compute_wi_uh_3(16, 99);break;case 13:call_onekernel_compute_wi_uh_3(17, 98);break;case 14:call_onekernel_compute_wi_uh_3(18, 97);break;case 15:call_onekernel_compute_wi_uh_3(19, 96);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_115(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(16, 99);break;case 1:call_onekernel_solve(17, 98);break;case 2:call_onekernel_solve(18, 97);break;case 3:call_onekernel_solve(19, 96);break;}
}