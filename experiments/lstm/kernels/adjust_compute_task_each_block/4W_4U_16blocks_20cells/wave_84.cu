#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 1)wave_compute_84(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi(0, 84);break;case 1:call_onekernel_compute_wi(1, 83);break;case 2:call_onekernel_compute_wi(2, 82);break;case 3:call_onekernel_compute_wi(3, 81);break;case 4:call_onekernel_compute_wi(4, 80);break;case 5:call_onekernel_compute_wi(5, 79);break;case 6:call_onekernel_compute_wi(6, 78);break;case 7:call_onekernel_compute_wi(7, 77);break;case 8:call_onekernel_compute_wi(8, 76);break;case 9:call_onekernel_compute_wi(9, 75);break;case 10:call_onekernel_compute_wi(10, 74);break;case 11:call_onekernel_compute_wi(11, 73);break;case 12:call_onekernel_compute_wi(12, 72);break;case 13:call_onekernel_compute_wi(13, 71);break;case 14:call_onekernel_compute_wi(14, 70);break;case 15:call_onekernel_compute_wi(15, 69);break;case 16:call_onekernel_compute_wi(16, 68);break;case 17:call_onekernel_compute_wi(17, 67);break;case 18:call_onekernel_compute_wi(18, 66);break;case 19:call_onekernel_compute_wi(19, 65);break;case 20:call_onekernel_compute_uh(0, 84);break;case 21:call_onekernel_compute_uh(1, 83);break;case 22:call_onekernel_compute_uh(2, 82);break;case 23:call_onekernel_compute_uh(3, 81);break;case 24:call_onekernel_compute_uh(4, 80);break;case 25:call_onekernel_compute_uh(5, 79);break;case 26:call_onekernel_compute_uh(6, 78);break;case 27:call_onekernel_compute_uh(7, 77);break;case 28:call_onekernel_compute_uh(8, 76);break;case 29:call_onekernel_compute_uh(9, 75);break;case 30:call_onekernel_compute_uh(10, 74);break;case 31:call_onekernel_compute_uh(11, 73);break;case 32:call_onekernel_compute_uh(12, 72);break;case 33:call_onekernel_compute_uh(13, 71);break;case 34:call_onekernel_compute_uh(14, 70);break;case 35:call_onekernel_compute_uh(15, 69);break;case 36:call_onekernel_compute_uh(16, 68);break;case 37:call_onekernel_compute_uh(17, 67);break;case 38:call_onekernel_compute_uh(18, 66);break;case 39:call_onekernel_compute_uh(19, 65);break;}
}__global__ void __launch_bounds__(256, 1)wave_solve_84(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 84);break;case 1:call_onekernel_solve(1, 83);break;case 2:call_onekernel_solve(2, 82);break;case 3:call_onekernel_solve(3, 81);break;case 4:call_onekernel_solve(4, 80);break;case 5:call_onekernel_solve(5, 79);break;case 6:call_onekernel_solve(6, 78);break;case 7:call_onekernel_solve(7, 77);break;case 8:call_onekernel_solve(8, 76);break;case 9:call_onekernel_solve(9, 75);break;case 10:call_onekernel_solve(10, 74);break;case 11:call_onekernel_solve(11, 73);break;case 12:call_onekernel_solve(12, 72);break;case 13:call_onekernel_solve(13, 71);break;case 14:call_onekernel_solve(14, 70);break;case 15:call_onekernel_solve(15, 69);break;case 16:call_onekernel_solve(16, 68);break;case 17:call_onekernel_solve(17, 67);break;case 18:call_onekernel_solve(18, 66);break;case 19:call_onekernel_solve(19, 65);break;}
}