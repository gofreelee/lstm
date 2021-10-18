#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_118(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(19, 99);break;case 1:call_onekernel_compute_wi_uh_1(19, 99);break;case 2:call_onekernel_compute_wi_uh_2(19, 99);break;case 3:call_onekernel_compute_wi_uh_3(19, 99);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_118(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(19, 99);break;}
}