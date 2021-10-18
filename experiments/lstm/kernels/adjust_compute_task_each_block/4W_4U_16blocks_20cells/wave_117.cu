#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_117(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(18, 99);break;case 1:call_onekernel_compute_wi(19, 98);break;case 2:call_onekernel_compute_uh(18, 99);break;case 3:call_onekernel_compute_uh(19, 98);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_117(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(18, 99);break;case 1:call_onekernel_solve(19, 98);break;}
}