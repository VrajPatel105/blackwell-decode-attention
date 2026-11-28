cmake_minimum_required(VERSION 3.24)
project(decode_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB KERNEL_SRCS CONFIGURE_DEPENDS src/kernels/*.cu)

add_executable(decode
  src/harness/main.cu
  src/harness/io.cu
  src/harness/compare.cu
  src/harness/timing.cu
  src/harness/ceiling.cu
  ${KERNEL_SRCS})
target_include_directories(decode PRIVATE include)
target_compile_options(decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
