#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_0(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_0(0, 0);break;case 1:call_onekernel_compute_wi_1(0, 0);break;case 2:call_onekernel_compute_wi_2(0, 0);break;case 3:call_onekernel_compute_wi_3(0, 0);break;case 4:call_onekernel_compute_uh_0(0, 0);break;case 5:call_onekernel_compute_uh_1(0, 0);break;case 6:call_onekernel_compute_uh_2(0, 0);break;case 7:call_onekernel_compute_uh_3(0, 0);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_0(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 0);break;}
}