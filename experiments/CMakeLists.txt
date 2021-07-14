cmake_minimum_required(VERSION 3.14)

set(HUNTER_PACKAGES abseil GTest)

include(FetchContent)
FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

set(CMAKE_BUILD_TYPE_INIT Release)

project(mica_experiments LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

add_subdirectory(lstm)
add_subdirectory(resnet50)