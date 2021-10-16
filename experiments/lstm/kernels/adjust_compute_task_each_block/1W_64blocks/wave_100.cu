#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4) wave_compute_100(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_0(1, 99);break;case 1:call_onekernel_compute_wi_0(2, 98);break;case 2:call_onekernel_compute_wi_0(3, 97);break;case 3:call_onekernel_compute_wi_0(4, 96);break;case 4:call_onekernel_compute_wi_0(5, 95);break;case 5:call_onekernel_compute_wi_0(6, 94);break;case 6:call_onekernel_compute_wi_0(7, 93);break;case 7:call_onekernel_compute_wi_0(8, 92);break;case 8:call_onekernel_compute_wi_0(9, 91);break;case 9:call_onekernel_compute_wi_1(1, 99);break;case 10:call_onekernel_compute_wi_1(2, 98);break;case 11:call_onekernel_compute_wi_1(3, 97);break;case 12:call_onekernel_compute_wi_1(4, 96);break;case 13:call_onekernel_compute_wi_1(5, 95);break;case 14:call_onekernel_compute_wi_1(6, 94);break;case 15:call_onekernel_compute_wi_1(7, 93);break;case 16:call_onekernel_compute_wi_1(8, 92);break;case 17:call_onekernel_compute_wi_1(9, 91);break;case 18:call_onekernel_compute_wi_2(1, 99);break;case 19:call_onekernel_compute_wi_2(2, 98);break;case 20:call_onekernel_compute_wi_2(3, 97);break;case 21:call_onekernel_compute_wi_2(4, 96);break;case 22:call_onekernel_compute_wi_2(5, 95);break;case 23:call_onekernel_compute_wi_2(6, 94);break;case 24:call_onekernel_compute_wi_2(7, 93);break;case 25:call_onekernel_compute_wi_2(8, 92);break;case 26:call_onekernel_compute_wi_2(9, 91);break;case 27:call_onekernel_compute_wi_3(1, 99);break;case 28:call_onekernel_compute_wi_3(2, 98);break;case 29:call_onekernel_compute_wi_3(3, 97);break;case 30:call_onekernel_compute_wi_3(4, 96);break;case 31:call_onekernel_compute_wi_3(5, 95);break;case 32:call_onekernel_compute_wi_3(6, 94);break;case 33:call_onekernel_compute_wi_3(7, 93);break;case 34:call_onekernel_compute_wi_3(8, 92);break;case 35:call_onekernel_compute_wi_3(9, 91);break;case 36:call_onekernel_compute_uh_0(1, 99);break;case 37:call_onekernel_compute_uh_0(2, 98);break;case 38:call_onekernel_compute_uh_0(3, 97);break;case 39:call_onekernel_compute_uh_0(4, 96);break;case 40:call_onekernel_compute_uh_0(5, 95);break;case 41:call_onekernel_compute_uh_0(6, 94);break;case 42:call_onekernel_compute_uh_0(7, 93);break;case 43:call_onekernel_compute_uh_0(8, 92);break;case 44:call_onekernel_compute_uh_0(9, 91);break;case 45:call_onekernel_compute_uh_1(1, 99);break;case 46:call_onekernel_compute_uh_1(2, 98);break;case 47:call_onekernel_compute_uh_1(3, 97);break;case 48:call_onekernel_compute_uh_1(4, 96);break;case 49:call_onekernel_compute_uh_1(5, 95);break;case 50:call_onekernel_compute_uh_1(6, 94);break;case 51:call_onekernel_compute_uh_1(7, 93);break;case 52:call_onekernel_compute_uh_1(8, 92);break;case 53:call_onekernel_compute_uh_1(9, 91);break;case 54:call_onekernel_compute_uh_2(1, 99);break;case 55:call_onekernel_compute_uh_2(2, 98);break;case 56:call_onekernel_compute_uh_2(3, 97);break;case 57:call_onekernel_compute_uh_2(4, 96);break;case 58:call_onekernel_compute_uh_2(5, 95);break;case 59:call_onekernel_compute_uh_2(6, 94);break;case 60:call_onekernel_compute_uh_2(7, 93);break;case 61:call_onekernel_compute_uh_2(8, 92);break;case 62:call_onekernel_compute_uh_2(9, 91);break;case 63:call_onekernel_compute_uh_3(1, 99);break;case 64:call_onekernel_compute_uh_3(2, 98);break;case 65:call_onekernel_compute_uh_3(3, 97);break;case 66:call_onekernel_compute_uh_3(4, 96);break;case 67:call_onekernel_compute_uh_3(5, 95);break;case 68:call_onekernel_compute_uh_3(6, 94);break;case 69:call_onekernel_compute_uh_3(7, 93);break;case 70:call_onekernel_compute_uh_3(8, 92);break;case 71:call_onekernel_compute_uh_3(9, 91);break;}
}__global__ void __launch_bounds__(256, 4) wave_solve_100(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(1, 99);break;case 1:call_onekernel_solve(2, 98);break;case 2:call_onekernel_solve(3, 97);break;case 3:call_onekernel_solve(4, 96);break;case 4:call_onekernel_solve(5, 95);break;case 5:call_onekernel_solve(6, 94);break;case 6:call_onekernel_solve(7, 93);break;case 7:call_onekernel_solve(8, 92);break;case 8:call_onekernel_solve(9, 91);break;}
}