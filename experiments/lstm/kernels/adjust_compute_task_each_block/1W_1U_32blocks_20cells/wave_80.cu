#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_80(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_uh_0(0, 80);break;case 1:call_onekernel_compute_wi_uh_0(1, 79);break;case 2:call_onekernel_compute_wi_uh_0(2, 78);break;case 3:call_onekernel_compute_wi_uh_0(3, 77);break;case 4:call_onekernel_compute_wi_uh_0(4, 76);break;case 5:call_onekernel_compute_wi_uh_0(5, 75);break;case 6:call_onekernel_compute_wi_uh_0(6, 74);break;case 7:call_onekernel_compute_wi_uh_0(7, 73);break;case 8:call_onekernel_compute_wi_uh_0(8, 72);break;case 9:call_onekernel_compute_wi_uh_0(9, 71);break;case 10:call_onekernel_compute_wi_uh_0(10, 70);break;case 11:call_onekernel_compute_wi_uh_0(11, 69);break;case 12:call_onekernel_compute_wi_uh_0(12, 68);break;case 13:call_onekernel_compute_wi_uh_0(13, 67);break;case 14:call_onekernel_compute_wi_uh_0(14, 66);break;case 15:call_onekernel_compute_wi_uh_0(15, 65);break;case 16:call_onekernel_compute_wi_uh_0(16, 64);break;case 17:call_onekernel_compute_wi_uh_0(17, 63);break;case 18:call_onekernel_compute_wi_uh_0(18, 62);break;case 19:call_onekernel_compute_wi_uh_0(19, 61);break;case 20:call_onekernel_compute_wi_uh_1(0, 80);break;case 21:call_onekernel_compute_wi_uh_1(1, 79);break;case 22:call_onekernel_compute_wi_uh_1(2, 78);break;case 23:call_onekernel_compute_wi_uh_1(3, 77);break;case 24:call_onekernel_compute_wi_uh_1(4, 76);break;case 25:call_onekernel_compute_wi_uh_1(5, 75);break;case 26:call_onekernel_compute_wi_uh_1(6, 74);break;case 27:call_onekernel_compute_wi_uh_1(7, 73);break;case 28:call_onekernel_compute_wi_uh_1(8, 72);break;case 29:call_onekernel_compute_wi_uh_1(9, 71);break;case 30:call_onekernel_compute_wi_uh_1(10, 70);break;case 31:call_onekernel_compute_wi_uh_1(11, 69);break;case 32:call_onekernel_compute_wi_uh_1(12, 68);break;case 33:call_onekernel_compute_wi_uh_1(13, 67);break;case 34:call_onekernel_compute_wi_uh_1(14, 66);break;case 35:call_onekernel_compute_wi_uh_1(15, 65);break;case 36:call_onekernel_compute_wi_uh_1(16, 64);break;case 37:call_onekernel_compute_wi_uh_1(17, 63);break;case 38:call_onekernel_compute_wi_uh_1(18, 62);break;case 39:call_onekernel_compute_wi_uh_1(19, 61);break;case 40:call_onekernel_compute_wi_uh_2(0, 80);break;case 41:call_onekernel_compute_wi_uh_2(1, 79);break;case 42:call_onekernel_compute_wi_uh_2(2, 78);break;case 43:call_onekernel_compute_wi_uh_2(3, 77);break;case 44:call_onekernel_compute_wi_uh_2(4, 76);break;case 45:call_onekernel_compute_wi_uh_2(5, 75);break;case 46:call_onekernel_compute_wi_uh_2(6, 74);break;case 47:call_onekernel_compute_wi_uh_2(7, 73);break;case 48:call_onekernel_compute_wi_uh_2(8, 72);break;case 49:call_onekernel_compute_wi_uh_2(9, 71);break;case 50:call_onekernel_compute_wi_uh_2(10, 70);break;case 51:call_onekernel_compute_wi_uh_2(11, 69);break;case 52:call_onekernel_compute_wi_uh_2(12, 68);break;case 53:call_onekernel_compute_wi_uh_2(13, 67);break;case 54:call_onekernel_compute_wi_uh_2(14, 66);break;case 55:call_onekernel_compute_wi_uh_2(15, 65);break;case 56:call_onekernel_compute_wi_uh_2(16, 64);break;case 57:call_onekernel_compute_wi_uh_2(17, 63);break;case 58:call_onekernel_compute_wi_uh_2(18, 62);break;case 59:call_onekernel_compute_wi_uh_2(19, 61);break;case 60:call_onekernel_compute_wi_uh_3(0, 80);break;case 61:call_onekernel_compute_wi_uh_3(1, 79);break;case 62:call_onekernel_compute_wi_uh_3(2, 78);break;case 63:call_onekernel_compute_wi_uh_3(3, 77);break;case 64:call_onekernel_compute_wi_uh_3(4, 76);break;case 65:call_onekernel_compute_wi_uh_3(5, 75);break;case 66:call_onekernel_compute_wi_uh_3(6, 74);break;case 67:call_onekernel_compute_wi_uh_3(7, 73);break;case 68:call_onekernel_compute_wi_uh_3(8, 72);break;case 69:call_onekernel_compute_wi_uh_3(9, 71);break;case 70:call_onekernel_compute_wi_uh_3(10, 70);break;case 71:call_onekernel_compute_wi_uh_3(11, 69);break;case 72:call_onekernel_compute_wi_uh_3(12, 68);break;case 73:call_onekernel_compute_wi_uh_3(13, 67);break;case 74:call_onekernel_compute_wi_uh_3(14, 66);break;case 75:call_onekernel_compute_wi_uh_3(15, 65);break;case 76:call_onekernel_compute_wi_uh_3(16, 64);break;case 77:call_onekernel_compute_wi_uh_3(17, 63);break;case 78:call_onekernel_compute_wi_uh_3(18, 62);break;case 79:call_onekernel_compute_wi_uh_3(19, 61);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_80(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 80);break;case 1:call_onekernel_solve(1, 79);break;case 2:call_onekernel_solve(2, 78);break;case 3:call_onekernel_solve(3, 77);break;case 4:call_onekernel_solve(4, 76);break;case 5:call_onekernel_solve(5, 75);break;case 6:call_onekernel_solve(6, 74);break;case 7:call_onekernel_solve(7, 73);break;case 8:call_onekernel_solve(8, 72);break;case 9:call_onekernel_solve(9, 71);break;case 10:call_onekernel_solve(10, 70);break;case 11:call_onekernel_solve(11, 69);break;case 12:call_onekernel_solve(12, 68);break;case 13:call_onekernel_solve(13, 67);break;case 14:call_onekernel_solve(14, 66);break;case 15:call_onekernel_solve(15, 65);break;case 16:call_onekernel_solve(16, 64);break;case 17:call_onekernel_solve(17, 63);break;case 18:call_onekernel_solve(18, 62);break;case 19:call_onekernel_solve(19, 61);break;}
}