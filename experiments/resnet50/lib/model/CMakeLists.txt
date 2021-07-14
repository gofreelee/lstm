add_library(models STATIC resnet.cc alexnet.cc)

target_link_libraries(models PUBLIC cudart kernels)