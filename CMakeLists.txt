cmake_minimum_required(VERSION 3.20)

# CUDA 12.2 rejects GCC > 12 in host compilation mode.
# In WSL/Ubuntu environments where GCC 13 is default, prefer GCC 12 when available.
if(UNIX AND NOT APPLE)
    if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/g++-12")
        set(CMAKE_CXX_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "Preferred C++ compiler" FORCE)
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/gcc-12")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12" CACHE FILEPATH "Preferred CUDA host compiler" FORCE)
    endif()
endif()

project(sinter CUDA CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SINTER_CORE_SOURCES
    src/ir.cpp
)

set(SINTER_PASSES_SOURCES
    src/passes/fusion.cpp
    src/passes/tiling.cpp
    src/passes/layout.cpp
)

set(SINTER_CODEGEN_SOURCES
    src/codegen/cuda_kernel_gen.cpp
    src/codegen/cuda_launcher_gen.cpp
    src/codegen/ptx_compiler.cpp
)

set(SINTER_AUTOTUNER_SOURCES
    src/autotuner/cost_model.cpp
    src/autotuner/schedule_space.cpp
    src/autotuner/benchmarker.cpp
)

set(SINTER_RUNTIME_SOURCES
    src/host_runtime/tensor.cpp
    src/host_runtime/executor.cpp
    src/host_runtime/profiler.cpp
)

set(SINTER_BACKEND_SOURCES
    src/backend/cuda_device.cpp
)

# Core library
add_library(sinter_core
    ${SINTER_CORE_SOURCES}
    ${SINTER_PASSES_SOURCES}
    ${SINTER_CODEGEN_SOURCES}
    ${SINTER_AUTOTUNER_SOURCES}
    ${SINTER_RUNTIME_SOURCES}
    ${SINTER_BACKEND_SOURCES}
)

target_link_libraries(sinter_core
    CUDA::toolkit
)

# Main executable
add_executable(sinter_compiler src/main.cpp)
target_link_libraries(sinter_compiler sinter_core)

# Testing
enable_testing()

option(ENABLE_TESTS "Enable building tests" OFF)

if(ENABLE_TESTS)
    include(FetchContent)
    set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.cache/sinter_fetchcontent")
    file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}")
    if(WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    # Unit tests
    add_executable(ir_test test/unit/ir_test.cpp)
    target_link_libraries(ir_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(ir_test)

    add_executable(passes_test test/unit/passes_test.cpp)
    target_link_libraries(passes_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(passes_test)

    add_executable(codegen_test test/unit/codegen_test.cpp)
    target_link_libraries(codegen_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(codegen_test)

    add_executable(autotuner_test test/unit/autotuner_test.cpp)
    target_link_libraries(autotuner_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(autotuner_test)

    # Integration tests
    add_executable(e2e_cpu_matmul_test test/integration/e2e_cpu_matmul_test.cpp)
    target_link_libraries(e2e_cpu_matmul_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(e2e_cpu_matmul_test)

    add_executable(e2e_matmul_test test/integration/e2e_matmul_test.cpp)
    target_link_libraries(e2e_matmul_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(e2e_matmul_test)

    add_executable(e2e_attention_test test/integration/e2e_attention_test.cpp)
    target_link_libraries(e2e_attention_test PRIVATE sinter_core GTest::gtest_main)
    gtest_discover_tests(e2e_attention_test)

    # Benchmark executable
    add_executable(benchmark_suite test/integration/benchmark_suite.cpp)
    target_link_libraries(benchmark_suite sinter_core)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(sinter_core PRIVATE /W4 /WX)
    target_compile_options(sinter_compiler PRIVATE /W4 /WX)
else()
    target_compile_options(sinter_core PRIVATE -Wall -Wextra -Werror)
    target_compile_options(sinter_compiler PRIVATE -Wall -Wextra -Werror)
endif()

# CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_80")

# Output directories
set_target_properties(sinter_compiler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
