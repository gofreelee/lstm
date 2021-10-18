#include "LstmExperimentLib.h"
__global__ void __launch_bounds__(256, 4)wave_compute_34(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_compute_wi_0(0, 34);break;case 1:call_onekernel_compute_wi_0(1, 33);break;case 2:call_onekernel_compute_wi_0(2, 32);break;case 3:call_onekernel_compute_wi_0(3, 31);break;case 4:call_onekernel_compute_wi_0(4, 30);break;case 5:call_onekernel_compute_wi_0(5, 29);break;case 6:call_onekernel_compute_wi_0(6, 28);break;case 7:call_onekernel_compute_wi_0(7, 27);break;case 8:call_onekernel_compute_wi_0(8, 26);break;case 9:call_onekernel_compute_wi_0(9, 25);break;case 10:call_onekernel_compute_wi_0(10, 24);break;case 11:call_onekernel_compute_wi_0(11, 23);break;case 12:call_onekernel_compute_wi_0(12, 22);break;case 13:call_onekernel_compute_wi_0(13, 21);break;case 14:call_onekernel_compute_wi_0(14, 20);break;case 15:call_onekernel_compute_wi_0(15, 19);break;case 16:call_onekernel_compute_wi_0(16, 18);break;case 17:call_onekernel_compute_wi_0(17, 17);break;case 18:call_onekernel_compute_wi_0(18, 16);break;case 19:call_onekernel_compute_wi_0(19, 15);break;case 20:call_onekernel_compute_wi_1(0, 34);break;case 21:call_onekernel_compute_wi_1(1, 33);break;case 22:call_onekernel_compute_wi_1(2, 32);break;case 23:call_onekernel_compute_wi_1(3, 31);break;case 24:call_onekernel_compute_wi_1(4, 30);break;case 25:call_onekernel_compute_wi_1(5, 29);break;case 26:call_onekernel_compute_wi_1(6, 28);break;case 27:call_onekernel_compute_wi_1(7, 27);break;case 28:call_onekernel_compute_wi_1(8, 26);break;case 29:call_onekernel_compute_wi_1(9, 25);break;case 30:call_onekernel_compute_wi_1(10, 24);break;case 31:call_onekernel_compute_wi_1(11, 23);break;case 32:call_onekernel_compute_wi_1(12, 22);break;case 33:call_onekernel_compute_wi_1(13, 21);break;case 34:call_onekernel_compute_wi_1(14, 20);break;case 35:call_onekernel_compute_wi_1(15, 19);break;case 36:call_onekernel_compute_wi_1(16, 18);break;case 37:call_onekernel_compute_wi_1(17, 17);break;case 38:call_onekernel_compute_wi_1(18, 16);break;case 39:call_onekernel_compute_wi_1(19, 15);break;case 40:call_onekernel_compute_wi_2(0, 34);break;case 41:call_onekernel_compute_wi_2(1, 33);break;case 42:call_onekernel_compute_wi_2(2, 32);break;case 43:call_onekernel_compute_wi_2(3, 31);break;case 44:call_onekernel_compute_wi_2(4, 30);break;case 45:call_onekernel_compute_wi_2(5, 29);break;case 46:call_onekernel_compute_wi_2(6, 28);break;case 47:call_onekernel_compute_wi_2(7, 27);break;case 48:call_onekernel_compute_wi_2(8, 26);break;case 49:call_onekernel_compute_wi_2(9, 25);break;case 50:call_onekernel_compute_wi_2(10, 24);break;case 51:call_onekernel_compute_wi_2(11, 23);break;case 52:call_onekernel_compute_wi_2(12, 22);break;case 53:call_onekernel_compute_wi_2(13, 21);break;case 54:call_onekernel_compute_wi_2(14, 20);break;case 55:call_onekernel_compute_wi_2(15, 19);break;case 56:call_onekernel_compute_wi_2(16, 18);break;case 57:call_onekernel_compute_wi_2(17, 17);break;case 58:call_onekernel_compute_wi_2(18, 16);break;case 59:call_onekernel_compute_wi_2(19, 15);break;case 60:call_onekernel_compute_wi_3(0, 34);break;case 61:call_onekernel_compute_wi_3(1, 33);break;case 62:call_onekernel_compute_wi_3(2, 32);break;case 63:call_onekernel_compute_wi_3(3, 31);break;case 64:call_onekernel_compute_wi_3(4, 30);break;case 65:call_onekernel_compute_wi_3(5, 29);break;case 66:call_onekernel_compute_wi_3(6, 28);break;case 67:call_onekernel_compute_wi_3(7, 27);break;case 68:call_onekernel_compute_wi_3(8, 26);break;case 69:call_onekernel_compute_wi_3(9, 25);break;case 70:call_onekernel_compute_wi_3(10, 24);break;case 71:call_onekernel_compute_wi_3(11, 23);break;case 72:call_onekernel_compute_wi_3(12, 22);break;case 73:call_onekernel_compute_wi_3(13, 21);break;case 74:call_onekernel_compute_wi_3(14, 20);break;case 75:call_onekernel_compute_wi_3(15, 19);break;case 76:call_onekernel_compute_wi_3(16, 18);break;case 77:call_onekernel_compute_wi_3(17, 17);break;case 78:call_onekernel_compute_wi_3(18, 16);break;case 79:call_onekernel_compute_wi_3(19, 15);break;case 80:call_onekernel_compute_uh_0(0, 34);break;case 81:call_onekernel_compute_uh_0(1, 33);break;case 82:call_onekernel_compute_uh_0(2, 32);break;case 83:call_onekernel_compute_uh_0(3, 31);break;case 84:call_onekernel_compute_uh_0(4, 30);break;case 85:call_onekernel_compute_uh_0(5, 29);break;case 86:call_onekernel_compute_uh_0(6, 28);break;case 87:call_onekernel_compute_uh_0(7, 27);break;case 88:call_onekernel_compute_uh_0(8, 26);break;case 89:call_onekernel_compute_uh_0(9, 25);break;case 90:call_onekernel_compute_uh_0(10, 24);break;case 91:call_onekernel_compute_uh_0(11, 23);break;case 92:call_onekernel_compute_uh_0(12, 22);break;case 93:call_onekernel_compute_uh_0(13, 21);break;case 94:call_onekernel_compute_uh_0(14, 20);break;case 95:call_onekernel_compute_uh_0(15, 19);break;case 96:call_onekernel_compute_uh_0(16, 18);break;case 97:call_onekernel_compute_uh_0(17, 17);break;case 98:call_onekernel_compute_uh_0(18, 16);break;case 99:call_onekernel_compute_uh_0(19, 15);break;case 100:call_onekernel_compute_uh_1(0, 34);break;case 101:call_onekernel_compute_uh_1(1, 33);break;case 102:call_onekernel_compute_uh_1(2, 32);break;case 103:call_onekernel_compute_uh_1(3, 31);break;case 104:call_onekernel_compute_uh_1(4, 30);break;case 105:call_onekernel_compute_uh_1(5, 29);break;case 106:call_onekernel_compute_uh_1(6, 28);break;case 107:call_onekernel_compute_uh_1(7, 27);break;case 108:call_onekernel_compute_uh_1(8, 26);break;case 109:call_onekernel_compute_uh_1(9, 25);break;case 110:call_onekernel_compute_uh_1(10, 24);break;case 111:call_onekernel_compute_uh_1(11, 23);break;case 112:call_onekernel_compute_uh_1(12, 22);break;case 113:call_onekernel_compute_uh_1(13, 21);break;case 114:call_onekernel_compute_uh_1(14, 20);break;case 115:call_onekernel_compute_uh_1(15, 19);break;case 116:call_onekernel_compute_uh_1(16, 18);break;case 117:call_onekernel_compute_uh_1(17, 17);break;case 118:call_onekernel_compute_uh_1(18, 16);break;case 119:call_onekernel_compute_uh_1(19, 15);break;case 120:call_onekernel_compute_uh_2(0, 34);break;case 121:call_onekernel_compute_uh_2(1, 33);break;case 122:call_onekernel_compute_uh_2(2, 32);break;case 123:call_onekernel_compute_uh_2(3, 31);break;case 124:call_onekernel_compute_uh_2(4, 30);break;case 125:call_onekernel_compute_uh_2(5, 29);break;case 126:call_onekernel_compute_uh_2(6, 28);break;case 127:call_onekernel_compute_uh_2(7, 27);break;case 128:call_onekernel_compute_uh_2(8, 26);break;case 129:call_onekernel_compute_uh_2(9, 25);break;case 130:call_onekernel_compute_uh_2(10, 24);break;case 131:call_onekernel_compute_uh_2(11, 23);break;case 132:call_onekernel_compute_uh_2(12, 22);break;case 133:call_onekernel_compute_uh_2(13, 21);break;case 134:call_onekernel_compute_uh_2(14, 20);break;case 135:call_onekernel_compute_uh_2(15, 19);break;case 136:call_onekernel_compute_uh_2(16, 18);break;case 137:call_onekernel_compute_uh_2(17, 17);break;case 138:call_onekernel_compute_uh_2(18, 16);break;case 139:call_onekernel_compute_uh_2(19, 15);break;case 140:call_onekernel_compute_uh_3(0, 34);break;case 141:call_onekernel_compute_uh_3(1, 33);break;case 142:call_onekernel_compute_uh_3(2, 32);break;case 143:call_onekernel_compute_uh_3(3, 31);break;case 144:call_onekernel_compute_uh_3(4, 30);break;case 145:call_onekernel_compute_uh_3(5, 29);break;case 146:call_onekernel_compute_uh_3(6, 28);break;case 147:call_onekernel_compute_uh_3(7, 27);break;case 148:call_onekernel_compute_uh_3(8, 26);break;case 149:call_onekernel_compute_uh_3(9, 25);break;case 150:call_onekernel_compute_uh_3(10, 24);break;case 151:call_onekernel_compute_uh_3(11, 23);break;case 152:call_onekernel_compute_uh_3(12, 22);break;case 153:call_onekernel_compute_uh_3(13, 21);break;case 154:call_onekernel_compute_uh_3(14, 20);break;case 155:call_onekernel_compute_uh_3(15, 19);break;case 156:call_onekernel_compute_uh_3(16, 18);break;case 157:call_onekernel_compute_uh_3(17, 17);break;case 158:call_onekernel_compute_uh_3(18, 16);break;case 159:call_onekernel_compute_uh_3(19, 15);break;}
}__global__ void __launch_bounds__(256, 4)wave_solve_34(WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output){switch (blockIdx.x >> 3) {
case 0:call_onekernel_solve(0, 34);break;case 1:call_onekernel_solve(1, 33);break;case 2:call_onekernel_solve(2, 32);break;case 3:call_onekernel_solve(3, 31);break;case 4:call_onekernel_solve(4, 30);break;case 5:call_onekernel_solve(5, 29);break;case 6:call_onekernel_solve(6, 28);break;case 7:call_onekernel_solve(7, 27);break;case 8:call_onekernel_solve(8, 26);break;case 9:call_onekernel_solve(9, 25);break;case 10:call_onekernel_solve(10, 24);break;case 11:call_onekernel_solve(11, 23);break;case 12:call_onekernel_solve(12, 22);break;case 13:call_onekernel_solve(13, 21);break;case 14:call_onekernel_solve(14, 20);break;case 15:call_onekernel_solve(15, 19);break;case 16:call_onekernel_solve(16, 18);break;case 17:call_onekernel_solve(17, 17);break;case 18:call_onekernel_solve(18, 16);break;case 19:call_onekernel_solve(19, 15);break;}
}