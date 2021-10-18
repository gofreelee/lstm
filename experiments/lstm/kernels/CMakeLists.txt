set(CMAKE_CUDA_ARCHITECTURES "80")



# file(GLOB IctLstm100_10_256_lib IctLstm100_10_256_lib/*.cu)

file(GLOB 1W_64blocks_lib adjust_compute_task_each_block/1W_64blocks/*.cu)

file(GLOB 4W_4U_16blocks_lib adjust_compute_task_each_block/4W_4U_16blocks/*.cu)

file(GLOB 1W_1U_32blocks_lib adjust_compute_task_each_block/1W_1U_32blocks/*.cu)

file(GLOB 2W_2U_16blocks_lib adjust_compute_task_each_block/2W_2U_16blocks/*.cu)

file(GLOB IctLstm100_10_256_NaiveFuse_FusedSolve_lib ictlstm100_10_256_naivefuse_fusedsolve_lib/*.cu)

file(GLOB IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_lib ictlstm100_10_256_naivefuse_fusedsolve_fusedcompute_lib/*.cu)

file(GLOB IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_lib ictlstm100_10_256_naivefuse_fusedsolve_fusedcompute_adduw_lib/*.cu) 

file(GLOB IctLstm100_10_256_Naivefuse_lib ictlstm100_10_256_naivefuse_lib/*.cu)

file(GLOB IctLstm100_10_256_Fusedcompute_lib ictlstm100_10_256_fusedcompute_lib/*.cu)

file(GLOB IctLstm100_10_256_Fusedcompute_128threads_lib ictlstm100_10_256_fusedcompute_128threads_lib/*.cu)

file(GLOB IctLstm100_10_256_Fusedcompute_512threads_lib ictlstm100_10_256_fusedcompute_512threads_lib/*.cu)

file(GLOB 1W_1U_32blocks_20cells_lib adjust_compute_task_each_block/1W_1U_32blocks_20cells/*.cu)

file(GLOB 1W_64blocks_20cells_lib adjust_compute_task_each_block/1W_64blocks_20cells/*.cu)

file(GLOB 4W_4U_16blocks_20cells_lib adjust_compute_task_each_block/4W_4U_16blocks_20cells/*.cu)

file(GLOB 8W_8blocks_20cells_lib adjust_compute_task_each_block/8W_8blocks_20cells/*.cu)

# cuda_add_library(IctLstmCellBS4
#     IctLstmCellBS4.cu)

# cuda_add_library(IctLstmCellBS16
#     IctLstmCellBS16.cu)

# cuda_add_library(IctLstmCell_cudalib
#     IctLstmCell.cu)

# cuda_add_library(IctLstm100_10_256_cudalib
#     ${IctLstm100_10_256_lib})

cuda_add_library(LstmExperiment4W_4U_16blocks_lib
    ${4W_4U_16blocks_lib})

cuda_add_library(LstmExperiment1W_64blocks_lib
    ${1W_64blocks_lib})

cuda_add_library(LstmExperiment1W_1U_32blocks_lib
    ${1W_1U_32blocks_lib})

cuda_add_library(LstmExperiment2W_2U_16blocks_lib
    ${2W_2U_16blocks_lib})

cuda_add_library(LstmExperiment1W_1U_32blocks_20cells_lib
    ${1W_1U_32blocks_20cells_lib})

cuda_add_library(LstmExperiment1W_64blocks_20cells_lib
    ${1W_64blocks_20cells_lib})

cuda_add_library(LstmExperiment4W_4U_16blocks_20cells_lib
    ${4W_4U_16blocks_20cells_lib}
)

cuda_add_library(LstmExperiment8W_8blocks_20cells_lib
    ${8W_8blocks_20cells_lib}
)

cuda_add_library(IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_cudalib
    ${IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_lib})

cuda_add_library(IctLstm100_10_256_cudalib
    ${IctLstm100_10_256_NaiveFuse_FusedSolve_lib})

cuda_add_library(IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_cudalib
    ${IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_lib})

cuda_add_library(IctLstm100_10_256_Naivefuse_cudalib
    ${IctLstm100_10_256_Naivefuse_lib})

cuda_add_library(IctLstm100_10_256_Fusedcompute_cudalib
    ${IctLstm100_10_256_Fusedcompute_lib})

cuda_add_library(IctLstm100_10_256_Fusedcompute_128threads_cudalib
    ${IctLstm100_10_256_Fusedcompute_128threads_lib}
)

cuda_add_library(IctLstm100_10_256_Fusedcompute_512threads_cudalib
    ${IctLstm100_10_256_Fusedcompute_512threads_lib}
)


