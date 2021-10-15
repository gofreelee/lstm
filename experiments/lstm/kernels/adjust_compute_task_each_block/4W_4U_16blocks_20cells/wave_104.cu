#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_104(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(5, 99);break;case 1:call_onekernel_compute_wi(6, 98);break;case 2:call_onekernel_compute_wi(7, 97);break;case 3:call_onekernel_compute_wi(8, 96);break;case 4:call_onekernel_compute_wi(9, 95);break;case 5:call_onekernel_compute_wi(10, 94);break;case 6:call_onekernel_compute_wi(11, 93);break;case 7:call_onekernel_compute_wi(12, 92);break;case 8:call_onekernel_compute_wi(13, 91);break;case 9:call_onekernel_compute_wi(14, 90);break;case 10:call_onekernel_compute_wi(15, 89);break;case 11:call_onekernel_compute_wi(16, 88);break;case 12:call_onekernel_compute_wi(17, 87);break;case 13:call_onekernel_compute_wi(18, 86);break;case 14:call_onekernel_compute_wi(19, 85);break;case 15:call_onekernel_compute_uh(5, 99);break;case 16:call_onekernel_compute_uh(6, 98);break;case 17:call_onekernel_compute_uh(7, 97);break;case 18:call_onekernel_compute_uh(8, 96);break;case 19:call_onekernel_compute_uh(9, 95);break;case 20:call_onekernel_compute_uh(10, 94);break;case 21:call_onekernel_compute_uh(11, 93);break;case 22:call_onekernel_compute_uh(12, 92);break;case 23:call_onekernel_compute_uh(13, 91);break;case 24:call_onekernel_compute_uh(14, 90);break;case 25:call_onekernel_compute_uh(15, 89);break;case 26:call_onekernel_compute_uh(16, 88);break;case 27:call_onekernel_compute_uh(17, 87);break;case 28:call_onekernel_compute_uh(18, 86);break;case 29:call_onekernel_compute_uh(19, 85);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_104(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(5, 99);break;case 1:call_onekernel_solve(6, 98);break;case 2:call_onekernel_solve(7, 97);break;case 3:call_onekernel_solve(8, 96);break;case 4:call_onekernel_solve(9, 95);break;case 5:call_onekernel_solve(10, 94);break;case 6:call_onekernel_solve(11, 93);break;case 7:call_onekernel_solve(12, 92);break;case 8:call_onekernel_solve(13, 91);break;case 9:call_onekernel_solve(14, 90);break;case 10:call_onekernel_solve(15, 89);break;case 11:call_onekernel_solve(16, 88);break;case 12:call_onekernel_solve(17, 87);break;case 13:call_onekernel_solve(18, 86);break;case 14:call_onekernel_solve(19, 85);break;}
}