cmake_minimum_required(VERSION 3.11)
project(mica_dataplane_gpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(shim)

install(TARGETS dataplane_gpu_shim DESTINATION .)