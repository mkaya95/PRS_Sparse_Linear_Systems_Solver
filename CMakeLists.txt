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
  src/formats/csr.cpp
  src/io/generators.cpp
  src/io/mtx_reader.cpp
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
