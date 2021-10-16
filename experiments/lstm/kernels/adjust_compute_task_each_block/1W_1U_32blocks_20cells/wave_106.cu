#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_106(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(7, 99);break;case 1:call_onekernel_compute_wi_uh_0(8, 98);break;case 2:call_onekernel_compute_wi_uh_0(9, 97);break;case 3:call_onekernel_compute_wi_uh_0(10, 96);break;case 4:call_onekernel_compute_wi_uh_0(11, 95);break;case 5:call_onekernel_compute_wi_uh_0(12, 94);break;case 6:call_onekernel_compute_wi_uh_0(13, 93);break;case 7:call_onekernel_compute_wi_uh_0(14, 92);break;case 8:call_onekernel_compute_wi_uh_0(15, 91);break;case 9:call_onekernel_compute_wi_uh_0(16, 90);break;case 10:call_onekernel_compute_wi_uh_0(17, 89);break;case 11:call_onekernel_compute_wi_uh_0(18, 88);break;case 12:call_onekernel_compute_wi_uh_0(19, 87);break;case 13:call_onekernel_compute_wi_uh_1(7, 99);break;case 14:call_onekernel_compute_wi_uh_1(8, 98);break;case 15:call_onekernel_compute_wi_uh_1(9, 97);break;case 16:call_onekernel_compute_wi_uh_1(10, 96);break;case 17:call_onekernel_compute_wi_uh_1(11, 95);break;case 18:call_onekernel_compute_wi_uh_1(12, 94);break;case 19:call_onekernel_compute_wi_uh_1(13, 93);break;case 20:call_onekernel_compute_wi_uh_1(14, 92);break;case 21:call_onekernel_compute_wi_uh_1(15, 91);break;case 22:call_onekernel_compute_wi_uh_1(16, 90);break;case 23:call_onekernel_compute_wi_uh_1(17, 89);break;case 24:call_onekernel_compute_wi_uh_1(18, 88);break;case 25:call_onekernel_compute_wi_uh_1(19, 87);break;case 26:call_onekernel_compute_wi_uh_2(7, 99);break;case 27:call_onekernel_compute_wi_uh_2(8, 98);break;case 28:call_onekernel_compute_wi_uh_2(9, 97);break;case 29:call_onekernel_compute_wi_uh_2(10, 96);break;case 30:call_onekernel_compute_wi_uh_2(11, 95);break;case 31:call_onekernel_compute_wi_uh_2(12, 94);break;case 32:call_onekernel_compute_wi_uh_2(13, 93);break;case 33:call_onekernel_compute_wi_uh_2(14, 92);break;case 34:call_onekernel_compute_wi_uh_2(15, 91);break;case 35:call_onekernel_compute_wi_uh_2(16, 90);break;case 36:call_onekernel_compute_wi_uh_2(17, 89);break;case 37:call_onekernel_compute_wi_uh_2(18, 88);break;case 38:call_onekernel_compute_wi_uh_2(19, 87);break;case 39:call_onekernel_compute_wi_uh_3(7, 99);break;case 40:call_onekernel_compute_wi_uh_3(8, 98);break;case 41:call_onekernel_compute_wi_uh_3(9, 97);break;case 42:call_onekernel_compute_wi_uh_3(10, 96);break;case 43:call_onekernel_compute_wi_uh_3(11, 95);break;case 44:call_onekernel_compute_wi_uh_3(12, 94);break;case 45:call_onekernel_compute_wi_uh_3(13, 93);break;case 46:call_onekernel_compute_wi_uh_3(14, 92);break;case 47:call_onekernel_compute_wi_uh_3(15, 91);break;case 48:call_onekernel_compute_wi_uh_3(16, 90);break;case 49:call_onekernel_compute_wi_uh_3(17, 89);break;case 50:call_onekernel_compute_wi_uh_3(18, 88);break;case 51:call_onekernel_compute_wi_uh_3(19, 87);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_106(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(7, 99);break;case 1:call_onekernel_solve(8, 98);break;case 2:call_onekernel_solve(9, 97);break;case 3:call_onekernel_solve(10, 96);break;case 4:call_onekernel_solve(11, 95);break;case 5:call_onekernel_solve(12, 94);break;case 6:call_onekernel_solve(13, 93);break;case 7:call_onekernel_solve(14, 92);break;case 8:call_onekernel_solve(15, 91);break;case 9:call_onekernel_solve(16, 90);break;case 10:call_onekernel_solve(17, 89);break;case 11:call_onekernel_solve(18, 88);break;case 12:call_onekernel_solve(19, 87);break;}
}