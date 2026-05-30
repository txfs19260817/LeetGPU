cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

project(LeetGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(LEETGPU_BUILD_TESTS "Build CUDA correctness tests" ON)
option(LEETGPU_BUILD_BENCHMARKS "Build NVBench benchmark executables" ON)

include(FetchContent)

find_package(CUDAToolkit REQUIRED)

if(LEETGPU_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)

  find_package(GTest CONFIG QUIET)
  if(NOT TARGET GTest::gtest_main)
    find_package(GTest QUIET)
  endif()

  if(NOT TARGET GTest::gtest_main AND TARGET GTest::Main)
    add_library(GTest::gtest_main INTERFACE IMPORTED)
    target_link_libraries(GTest::gtest_main INTERFACE GTest::Main)
  endif()

  if(NOT TARGET GTest::gtest_main)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
  endif()
endif()

if(LEETGPU_BUILD_BENCHMARKS)
  set(NVBench_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(NVBench_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
  set(NVBench_ENABLE_DEVICE_TESTING OFF CACHE BOOL "" FORCE)
  set(NVBench_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    NVBench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG 7ba2b79d5b08194b3448ff909a55d0a5baacdbb0
  )
  FetchContent_MakeAvailable(NVBench)
endif()

function(add_cuda_exercise exercise_dir)
  get_filename_component(exercise_name "${exercise_dir}" NAME)
  string(REPLACE "-" "_" target_prefix "${exercise_name}")

  set(src_cuda "${exercise_dir}/src_cuda.cu")
  set(test_cuda "${exercise_dir}/test_cuda.cu")
  set(bench_cuda "${exercise_dir}/benchmark_cuda.cu")

  add_library("${target_prefix}_cuda" STATIC "${src_cuda}")
  target_include_directories("${target_prefix}_cuda" PUBLIC "${exercise_dir}")
  target_link_libraries("${target_prefix}_cuda" PUBLIC CUDA::cudart)

  if(LEETGPU_BUILD_TESTS AND EXISTS "${test_cuda}")
    add_executable("${target_prefix}_test" "${test_cuda}")
    target_link_libraries("${target_prefix}_test" PRIVATE "${target_prefix}_cuda" GTest::gtest_main)
    gtest_discover_tests(
      "${target_prefix}_test"
      DISCOVERY_TIMEOUT 60
      TEST_PREFIX "${target_prefix}."
    )
  endif()

  if(LEETGPU_BUILD_BENCHMARKS AND EXISTS "${bench_cuda}")
    add_executable("${target_prefix}_benchmark" "${bench_cuda}")
    target_link_libraries("${target_prefix}_benchmark" PRIVATE "${target_prefix}_cuda" nvbench::main)
  endif()
endfunction()

file(GLOB exercise_sources CONFIGURE_DEPENDS RELATIVE "${CMAKE_SOURCE_DIR}" "[0-9][0-9][0-9]*-*/src_cuda.cu")

foreach(exercise_source IN LISTS exercise_sources)
  get_filename_component(exercise_dir "${CMAKE_SOURCE_DIR}/${exercise_source}" DIRECTORY)
  add_cuda_exercise("${exercise_dir}")
endforeach()
