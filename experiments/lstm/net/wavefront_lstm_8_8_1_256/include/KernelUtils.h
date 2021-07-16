#ifndef _LSTM_KERNELUTILS_H_
#define _LSTM_KERNELUTILS_H_

#include "RammerLikeArgs.h"

void initConstant2(int numOfCells, const float *host_bias[]);

void initConstant3(int numOfCells, const float *host_bias[]);

template <unsigned int t_num_layer, unsigned int t_hidden_size>
void bigKernel(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
               RammerLikeCellModel<t_hidden_size> *__restrict__ models,
               RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernelPreZero(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel0(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel1(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel2(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel3(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel4(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel5(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel6(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel7(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel8(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel9(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
             RammerLikeCellModel<t_hidden_size> *__restrict__ model,
             RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel10(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
              RammerLikeCellModel<t_hidden_size> *__restrict__ model,
              RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel11(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
              RammerLikeCellModel<t_hidden_size> *__restrict__ model,
              RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel12(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
              RammerLikeCellModel<t_hidden_size> *__restrict__ model,
              RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel13(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
              RammerLikeCellModel<t_hidden_size> *__restrict__ model,
              RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void kernel14(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
              RammerLikeCellModel<t_hidden_size> *__restrict__ model,
              RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, int t_cell, bool update_state_c = true>
void ictOneKernelRowFirstWrapper(
    RammerLikeCellInput<t_hidden_size> *__restrict__ input,
    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel0(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel1(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel2(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel3(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel4(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel5(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel6(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel7(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel8(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel9(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                   RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                   RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel10(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel11(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel12(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel13(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void MatMulKernel14(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                    RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel0(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel1(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel2(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel3(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel4(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel5(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel6(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel7(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel8(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel9(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel10(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                 RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                 RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel11(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                 RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                 RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel12(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                 RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                 RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel13(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                 RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                 RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
void P2PKernel14(RammerLikeCellInput<t_hidden_size> *__restrict__ input,
                 RammerLikeCellModel<t_hidden_size> *__restrict__ model,
                 RammerLikeCellOutput *__restrict__ output);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok_1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok0(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok2(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok3(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok4(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok5(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok6(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok7(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok8(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok9(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                    RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                    RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok10(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok11(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);
template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok15(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok16(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok17(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok18(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok19(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok20(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok21(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok22(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok23(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok24(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok25(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok26(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok27(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok28(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok29(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok30(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok31(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok32(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok33(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok34(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok35(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok36(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok37(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok38(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok39(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok40(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok41(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok42(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok43(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok44(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok45(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok46(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok47(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok48(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok49(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok50(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok51(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok52(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok53(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok54(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok55(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok56(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok57(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok58(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok59(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok60(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok61(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok62(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok63(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok64(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok65(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok66(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok67(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok68(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok69(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok70(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok71(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok72(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok73(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok74(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok75(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok76(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok77(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok78(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok79(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok80(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok81(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok82(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok83(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok84(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok85(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok86(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok87(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok88(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok89(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok90(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok91(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok92(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok93(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok94(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok95(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok96(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok97(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok98(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok99(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok100(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok101(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok102(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok103(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok104(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok105(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok106(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok107(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok108(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok109(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                      RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                      RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok12(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok13(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void ok14(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
                     RammerLikeCellModel<t_hidden_size> *__restrict__ models,
                     RammerLikeCellOutput *__restrict__ outputs);

#endif