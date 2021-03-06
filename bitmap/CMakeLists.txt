cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(bitmap CXX CUDA)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

set(NVCC_FLAGS --expt-extended-lambda -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH})

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
#cuda requirements
include_directories(/opt/cuda/include)

set(ssjoinfiles main.cxx input.cxx host_timing.cpp classes.hxx)

add_executable(bitmap ${ssjoinfiles})
set_property(TARGET bitmap PROPERTY CXX_STANDARD 11)
target_compile_options(bitmap PRIVATE -O3)

add_library(device STATIC
		gpu.hxx structs.hxx bitmap.hxx verification.hxx  utils.hxx handler.hxx device_timing.hxx
		 gpu.cu handler.cu device_timing.cu
		)

set_property(TARGET device PROPERTY CUDA_STANDARD 11)
target_compile_options(device PRIVATE ${NVCC_FLAGS})

target_link_libraries(bitmap device)
