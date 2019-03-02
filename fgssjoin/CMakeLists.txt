cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(fgssjoin CXX CUDA)

find_package(OpenMP REQUIRED)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

set(NVCC_FLAGS --expt-extended-lambda -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH})

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
#cuda requirements
include_directories(/opt/cuda/include)

set(ssjoinfiles main.cpp)

add_executable(fgssjoin ${ssjoinfiles})
set_property(TARGET fgssjoin PROPERTY CXX_STANDARD 11)
target_compile_options(fgssjoin PRIVATE -O3 -Wall ${OpenMP_CXX_FLAGS})

add_library(device STATIC
        inverted_index.cuh inverted_index.cu main.cu simjoin.cu simjoin.cuh structs.cuh utils.cu utils.cuh device_timing.hxx device_timing.cu
        )

set_property(TARGET device PROPERTY CUDA_STANDARD 11)
target_compile_options(device PRIVATE ${NVCC_FLAGS})

target_link_libraries(fgssjoin device ${OpenMP_CXX_FLAGS})
