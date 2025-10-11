cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)
project(LeetGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

# ===== Dependencies =====
include(FetchContent)

find_package(CUDAToolkit REQUIRED)
# GTest
find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        main  # 或指定版本如 release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()
# Add libtorch
list(APPEND CMAKE_PREFIX_PATH ".venv/lib/python3.12/site-packages/torch")
find_package(Torch REQUIRED)
# Fetch NVBench
set(NVBench_ENABLE_TESTING OFF)
set(NVBench_ENABLE_HEADER_TESTING OFF)
set(NVBench_ENABLE_DEVICE_TESTING OFF)
set(NVBench_ENABLE_EXAMPLES OFF)
FetchContent_Declare(
    NVBench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG b88a45f4170af4e907e69af22a55af67859d3b49
)
FetchContent_MakeAvailable(NVBench)

# Locate all src_cuda.cu, test_cuda.cu and benchmark_cuda.cu files in a subdirectory
function(add_cuda_test TEST_NAME DIR)
    # Set source files
    set(SRC_CUDA   ${DIR}/src_cuda.cu)
    set(SRC_TEST   ${DIR}/test_cuda.cu)
    set(SRC_BENCH  ${DIR}/benchmark_cuda.cu)

    # Library (operator implementation) - create a library for the operator implementation
    add_library(${TEST_NAME}_lib ${SRC_CUDA})

    # Executable (test) - link to the library and GTest
    add_executable(${TEST_NAME} ${SRC_TEST})
    target_link_libraries(${TEST_NAME}
        PRIVATE ${TEST_NAME}_lib
        PRIVATE GTest::GTest GTest::Main
    )

    # Add test target by test granularity
    include(GoogleTest)
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        DISCOVERY_TIMEOUT 30
        TEST_PREFIX "${TEST_NAME}."
    )

    # Benchmark executable (if benchmark file exists)
    if(EXISTS ${SRC_BENCH})
        add_executable(${TEST_NAME}_benchmark ${SRC_BENCH})
        target_link_libraries(${TEST_NAME}_benchmark
            PRIVATE ${TEST_NAME}_lib
            PRIVATE nvbench::main
        )
    endif()
endfunction()

# Add all subdirectory tests (find all src_cuda.cu files in subdirectories)
file(GLOB SUBDIRS RELATIVE ${CMAKE_SOURCE_DIR} "*/src_cuda.cu")

foreach(CUDA_FILE ${SUBDIRS})
    get_filename_component(DIR ${CUDA_FILE} DIRECTORY)
    string(REPLACE "-" "_" TEST_NAME ${DIR})  # Normalize directory name to a valid target name
    add_cuda_test(${TEST_NAME} ${CMAKE_SOURCE_DIR}/${DIR})
endforeach()
