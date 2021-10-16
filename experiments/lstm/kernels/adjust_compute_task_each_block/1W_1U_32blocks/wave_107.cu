#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(8, 99);break;case 1:call_onekernel_compute_wi_uh_0(9, 98);break;case 2:call_onekernel_compute_wi_uh_1(8, 99);break;case 3:call_onekernel_compute_wi_uh_1(9, 98);break;case 4:call_onekernel_compute_wi_uh_2(8, 99);break;case 5:call_onekernel_compute_wi_uh_2(9, 98);break;case 6:call_onekernel_compute_wi_uh_3(8, 99);break;case 7:call_onekernel_compute_wi_uh_3(9, 98);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(8, 99);break;case 1:call_onekernel_solve(9, 98);break;}
}