cmake_minimum_required(VERSION 3.21)
project(ssolve LANGUAGES CXX)

if(MSVC)
  message(FATAL_ERROR "Use GCC or Clang on Linux, WSL2 or macOS. MSVC is not supported.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SSOLVE_NATIVE "Compile for this machine's CPU with -march=native" ON)
set(SSOLVE_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" CACHE PATH "Eigen checkout to benchmark against")

# Same flags for our code and Eigen, so comparisons are fair.
# -fopenmp-simd enables `omp simd` without threads or the OpenMP runtime.
add_compile_options(-Wall -Wextra -Wpedantic -fopenmp-simd)
if(SSOLVE_NATIVE)
  add_compile_options(-march=native)
endif()

include_directories(include)

enable_testing()

function(ssolve_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

ssolve_test(test_aligned_array)
