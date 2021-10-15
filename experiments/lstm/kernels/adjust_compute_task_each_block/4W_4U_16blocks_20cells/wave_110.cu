#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_110(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(11, 99);break;case 1:call_onekernel_compute_wi(12, 98);break;case 2:call_onekernel_compute_wi(13, 97);break;case 3:call_onekernel_compute_wi(14, 96);break;case 4:call_onekernel_compute_wi(15, 95);break;case 5:call_onekernel_compute_wi(16, 94);break;case 6:call_onekernel_compute_wi(17, 93);break;case 7:call_onekernel_compute_wi(18, 92);break;case 8:call_onekernel_compute_wi(19, 91);break;case 9:call_onekernel_compute_uh(11, 99);break;case 10:call_onekernel_compute_uh(12, 98);break;case 11:call_onekernel_compute_uh(13, 97);break;case 12:call_onekernel_compute_uh(14, 96);break;case 13:call_onekernel_compute_uh(15, 95);break;case 14:call_onekernel_compute_uh(16, 94);break;case 15:call_onekernel_compute_uh(17, 93);break;case 16:call_onekernel_compute_uh(18, 92);break;case 17:call_onekernel_compute_uh(19, 91);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_110(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(11, 99);break;case 1:call_onekernel_solve(12, 98);break;case 2:call_onekernel_solve(13, 97);break;case 3:call_onekernel_solve(14, 96);break;case 4:call_onekernel_solve(15, 95);break;case 5:call_onekernel_solve(16, 94);break;case 6:call_onekernel_solve(17, 93);break;case 7:call_onekernel_solve(18, 92);break;case 8:call_onekernel_solve(19, 91);break;}
}