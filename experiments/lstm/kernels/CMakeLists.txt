set(CMAKE_CUDA_ARCHITECTURES "80;86")

cuda_add_library(LstmComputeOneKernel
    LstmComputeOneKernel.cu)

cuda_add_library(LstmNaiveCuda
    LstmNaive.cu)

cuda_add_library(Wavefront_lstm_8_8_1_256_cudalib
    IctKernels.cu)

cuda_add_library(Wavefront_lstm_10_100_1_256_cudalib
    IctKernels_10_100.cu)

cuda_add_library(IctLstmCell_cudalib
    IctLstmCell.cu)

cuda_add_library(IctLstm100_10_256_cudalib
    IctLstm100_10_256_lib.cu)

cuda_add_library(IctSeq2SeqLstm_cudalib
    IctSeq2SeqLstmlib.cu)
