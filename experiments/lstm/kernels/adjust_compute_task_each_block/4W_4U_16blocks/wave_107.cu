#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(8, 99);break;case 1:call_onekernel_compute_wi(9, 98);break;case 2:call_onekernel_compute_uh(8, 99);break;case 3:call_onekernel_compute_uh(9, 98);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(8, 99);break;case 1:call_onekernel_solve(9, 98);break;}
}