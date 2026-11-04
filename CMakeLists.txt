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

# Build information for the benchmark manifest, refreshed on every build.
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}} -fopenmp-simd" build_flags)
if(SSOLVE_NATIVE)
  string(APPEND build_flags " -march=native")
endif()
set(build_info_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(build_info_header "${build_info_dir}/ssolve_build_info.h")
add_custom_target(ssolve_build_info
  BYPRODUCTS "${build_info_header}"
  COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${build_info_header}"
          "-DFLAGS=${build_flags}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_build_info.cmake"
  VERBATIM)

add_library(ssolve STATIC
  src/bench/csv.cpp
  src/bench/manifest.cpp
  src/backends/seq/seq_backend.cpp
  src/formats/csr.cpp
  src/io/generators.cpp
  src/io/mtx_reader.cpp
  src/solvers/cg.cpp
  src/verify.cpp
)
add_dependencies(ssolve ssolve_build_info)
set_source_files_properties(src/bench/manifest.cpp PROPERTIES INCLUDE_DIRECTORIES "${build_info_dir}")

# STREAM measures the machine's memory bandwidth, so it alone uses threads for now.
find_package(OpenMP)
add_executable(stream apps/stream.cpp)
target_link_libraries(stream PRIVATE ssolve)
if(OpenMP_CXX_FOUND)
  target_link_libraries(stream PRIVATE OpenMP::OpenMP_CXX)
endif()

# Benchmark fixtures: the Eigen baselines and the optimization ladder.
add_library(ssolve_bench STATIC
  bench/eigen_kernels.cpp
  bench/eigen_solvers.cpp
  bench/ladder.cpp
  bench/matrix_spec.cpp
  bench/registry.cpp
)
target_include_directories(ssolve_bench PUBLIC bench)
target_include_directories(ssolve_bench SYSTEM PRIVATE "${SSOLVE_EIGEN_DIR}")
target_link_libraries(ssolve_bench PUBLIC ssolve)

add_executable(bench_cg apps/bench_cg.cpp)
target_link_libraries(bench_cg PRIVATE ssolve_bench)
add_executable(bench_kernels apps/bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE ssolve_bench)

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
ssolve_test(test_csv)
ssolve_test(test_backend)

add_executable(test_solvers tests/test_solvers.cpp)
target_include_directories(test_solvers PRIVATE tests)
target_link_libraries(test_solvers PRIVATE ssolve_bench)
add_test(NAME test_solvers COMMAND test_solvers)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME test_scripts
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scripts.py" "$<TARGET_FILE_DIR:bench_cg>")
endif()
