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
# -fopenmp-simd enables `omp simd` without threads or the OpenMP runtime.
add_compile_options(-Wall -Wextra -Wpedantic -fopenmp-simd)
if(SSOLVE_NATIVE)
  add_compile_options(-march=native)
endif()

include_directories(include)

add_library(ssolve STATIC
  src/backends/seq/seq_backend.cpp
  src/formats/csr.cpp
  src/io/generators.cpp
  src/io/mtx_reader.cpp
  src/io/matrix_spec.cpp
  src/solvers/cg.cpp
  src/verify.cpp
)

# Compiler remarks showing which solver kernel loops were vectorized, and why others were not.
option(SSOLVE_VEC_REPORT "Print vectorization remarks while compiling the solver kernels" OFF)
if(SSOLVE_VEC_REPORT)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(vec_report_flags -fopt-info-vec-optimized -fopt-info-vec-missed)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(vec_report_flags -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize)
  endif()
  set_source_files_properties(src/backends/seq/seq_backend.cpp PROPERTIES COMPILE_OPTIONS "${vec_report_flags}")
endif()

add_executable(solve apps/solve.cpp)
target_link_libraries(solve PRIVATE ssolve)

enable_testing()

function(ssolve_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  target_link_libraries(${name} PRIVATE ssolve)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

ssolve_test(test_aligned_array)
ssolve_test(test_matrix)
ssolve_test(test_mtx_reader)
ssolve_test(test_generators)
ssolve_test(test_backend)
ssolve_test(test_solvers)
