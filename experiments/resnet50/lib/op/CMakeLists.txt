set(CMAKE_CUDA_ARCHITECTURES "80;86")

add_library(kernels STATIC cuda_ops.cu)