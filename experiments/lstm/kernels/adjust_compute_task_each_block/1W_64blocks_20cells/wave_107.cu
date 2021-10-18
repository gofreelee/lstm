#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_0(8, 99);break;case 1:call_onekernel_compute_wi_0(9, 98);break;case 2:call_onekernel_compute_wi_0(10, 97);break;case 3:call_onekernel_compute_wi_0(11, 96);break;case 4:call_onekernel_compute_wi_0(12, 95);break;case 5:call_onekernel_compute_wi_0(13, 94);break;case 6:call_onekernel_compute_wi_0(14, 93);break;case 7:call_onekernel_compute_wi_0(15, 92);break;case 8:call_onekernel_compute_wi_0(16, 91);break;case 9:call_onekernel_compute_wi_0(17, 90);break;case 10:call_onekernel_compute_wi_0(18, 89);break;case 11:call_onekernel_compute_wi_0(19, 88);break;case 12:call_onekernel_compute_wi_1(8, 99);break;case 13:call_onekernel_compute_wi_1(9, 98);break;case 14:call_onekernel_compute_wi_1(10, 97);break;case 15:call_onekernel_compute_wi_1(11, 96);break;case 16:call_onekernel_compute_wi_1(12, 95);break;case 17:call_onekernel_compute_wi_1(13, 94);break;case 18:call_onekernel_compute_wi_1(14, 93);break;case 19:call_onekernel_compute_wi_1(15, 92);break;case 20:call_onekernel_compute_wi_1(16, 91);break;case 21:call_onekernel_compute_wi_1(17, 90);break;case 22:call_onekernel_compute_wi_1(18, 89);break;case 23:call_onekernel_compute_wi_1(19, 88);break;case 24:call_onekernel_compute_wi_2(8, 99);break;case 25:call_onekernel_compute_wi_2(9, 98);break;case 26:call_onekernel_compute_wi_2(10, 97);break;case 27:call_onekernel_compute_wi_2(11, 96);break;case 28:call_onekernel_compute_wi_2(12, 95);break;case 29:call_onekernel_compute_wi_2(13, 94);break;case 30:call_onekernel_compute_wi_2(14, 93);break;case 31:call_onekernel_compute_wi_2(15, 92);break;case 32:call_onekernel_compute_wi_2(16, 91);break;case 33:call_onekernel_compute_wi_2(17, 90);break;case 34:call_onekernel_compute_wi_2(18, 89);break;case 35:call_onekernel_compute_wi_2(19, 88);break;case 36:call_onekernel_compute_wi_3(8, 99);break;case 37:call_onekernel_compute_wi_3(9, 98);break;case 38:call_onekernel_compute_wi_3(10, 97);break;case 39:call_onekernel_compute_wi_3(11, 96);break;case 40:call_onekernel_compute_wi_3(12, 95);break;case 41:call_onekernel_compute_wi_3(13, 94);break;case 42:call_onekernel_compute_wi_3(14, 93);break;case 43:call_onekernel_compute_wi_3(15, 92);break;case 44:call_onekernel_compute_wi_3(16, 91);break;case 45:call_onekernel_compute_wi_3(17, 90);break;case 46:call_onekernel_compute_wi_3(18, 89);break;case 47:call_onekernel_compute_wi_3(19, 88);break;case 48:call_onekernel_compute_uh_0(8, 99);break;case 49:call_onekernel_compute_uh_0(9, 98);break;case 50:call_onekernel_compute_uh_0(10, 97);break;case 51:call_onekernel_compute_uh_0(11, 96);break;case 52:call_onekernel_compute_uh_0(12, 95);break;case 53:call_onekernel_compute_uh_0(13, 94);break;case 54:call_onekernel_compute_uh_0(14, 93);break;case 55:call_onekernel_compute_uh_0(15, 92);break;case 56:call_onekernel_compute_uh_0(16, 91);break;case 57:call_onekernel_compute_uh_0(17, 90);break;case 58:call_onekernel_compute_uh_0(18, 89);break;case 59:call_onekernel_compute_uh_0(19, 88);break;case 60:call_onekernel_compute_uh_1(8, 99);break;case 61:call_onekernel_compute_uh_1(9, 98);break;case 62:call_onekernel_compute_uh_1(10, 97);break;case 63:call_onekernel_compute_uh_1(11, 96);break;case 64:call_onekernel_compute_uh_1(12, 95);break;case 65:call_onekernel_compute_uh_1(13, 94);break;case 66:call_onekernel_compute_uh_1(14, 93);break;case 67:call_onekernel_compute_uh_1(15, 92);break;case 68:call_onekernel_compute_uh_1(16, 91);break;case 69:call_onekernel_compute_uh_1(17, 90);break;case 70:call_onekernel_compute_uh_1(18, 89);break;case 71:call_onekernel_compute_uh_1(19, 88);break;case 72:call_onekernel_compute_uh_2(8, 99);break;case 73:call_onekernel_compute_uh_2(9, 98);break;case 74:call_onekernel_compute_uh_2(10, 97);break;case 75:call_onekernel_compute_uh_2(11, 96);break;case 76:call_onekernel_compute_uh_2(12, 95);break;case 77:call_onekernel_compute_uh_2(13, 94);break;case 78:call_onekernel_compute_uh_2(14, 93);break;case 79:call_onekernel_compute_uh_2(15, 92);break;case 80:call_onekernel_compute_uh_2(16, 91);break;case 81:call_onekernel_compute_uh_2(17, 90);break;case 82:call_onekernel_compute_uh_2(18, 89);break;case 83:call_onekernel_compute_uh_2(19, 88);break;case 84:call_onekernel_compute_uh_3(8, 99);break;case 85:call_onekernel_compute_uh_3(9, 98);break;case 86:call_onekernel_compute_uh_3(10, 97);break;case 87:call_onekernel_compute_uh_3(11, 96);break;case 88:call_onekernel_compute_uh_3(12, 95);break;case 89:call_onekernel_compute_uh_3(13, 94);break;case 90:call_onekernel_compute_uh_3(14, 93);break;case 91:call_onekernel_compute_uh_3(15, 92);break;case 92:call_onekernel_compute_uh_3(16, 91);break;case 93:call_onekernel_compute_uh_3(17, 90);break;case 94:call_onekernel_compute_uh_3(18, 89);break;case 95:call_onekernel_compute_uh_3(19, 88);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_107(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(8, 99);break;case 1:call_onekernel_solve(9, 98);break;case 2:call_onekernel_solve(10, 97);break;case 3:call_onekernel_solve(11, 96);break;case 4:call_onekernel_solve(12, 95);break;case 5:call_onekernel_solve(13, 94);break;case 6:call_onekernel_solve(14, 93);break;case 7:call_onekernel_solve(15, 92);break;case 8:call_onekernel_solve(16, 91);break;case 9:call_onekernel_solve(17, 90);break;case 10:call_onekernel_solve(18, 89);break;case 11:call_onekernel_solve(19, 88);break;}
}