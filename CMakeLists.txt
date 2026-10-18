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

# =============================================================================
# LibTorch Configuration
# =============================================================================
# LibTorch is used for benchmarking Sinter against PyTorch
# It provides torch::matmul, torch::add, etc. as reference implementations
#
# KNOWN ISSUE: Conda's PyTorch has CMake integration issues with FindCUDA.
# For now, LibTorch support is disabled by default. We'll benchmark against
# cuBLAS first (which works), then add LibTorch support later.
#
# To try enabling manually:
#   cmake -DENABLE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ...
# =============================================================================
option(ENABLE_LIBTORCH "Enable LibTorch for benchmarking (experimental)" OFF)

if(ENABLE_LIBTORCH)
    # Set CUDA paths for the old FindCUDA module that Caffe2 uses
    if(CUDAToolkit_FOUND)
        get_filename_component(_cuda_include_parent "${CUDAToolkit_INCLUDE_DIRS}" DIRECTORY)
        get_filename_component(_cuda_target_parent "${_cuda_include_parent}" DIRECTORY)
        get_filename_component(_cuda_root "${_cuda_target_parent}" DIRECTORY)
        
        set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_root}" CACHE PATH "CUDA Toolkit root" FORCE)
        set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}" CACHE PATH "CUDA include dirs" FORCE)
        
        message(STATUS "Setting CUDA_TOOLKIT_ROOT_DIR for Caffe2: ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()
    
    find_package(Torch QUIET)
    
    if(Torch_FOUND)
        message(STATUS "Found LibTorch: ${TORCH_INCLUDE_DIRS}")
        set(SINTER_HAS_LIBTORCH TRUE)
    else()
        message(WARNING "LibTorch not found or failed to configure.")
        set(SINTER_HAS_LIBTORCH FALSE)
    endif()
else()
    message(STATUS "LibTorch disabled. Will benchmark against cuBLAS only.")
    set(SINTER_HAS_LIBTORCH FALSE)
endif()

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
    src/codegen/ptx_compiler.cpp       # NVRTC-based compilation (CompiledKernel, NvrtcCompiler)
    src/codegen/cuda_launcher_gen.cpp  # Legacy stub (to be removed later)
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

# Find NVRTC library manually if CMake didn't find it
# This is needed for conda environments where NVRTC isn't in the standard CMake search paths
if(NOT TARGET CUDA::nvrtc)
    # Get the directory containing nvcc to find NVRTC in the same tree
    get_filename_component(_nvcc_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_cuda_prefix "${_nvcc_dir}" DIRECTORY)
    
    find_library(NVRTC_LIBRARY
        NAMES nvrtc nvrtc.12 libnvrtc.so.12
        HINTS 
            "${_cuda_prefix}/lib"              # Conda: $CONDA_PREFIX/lib
            "${_cuda_prefix}/lib64"            # Some setups use lib64
            "${CUDAToolkit_LIBRARY_DIR}"       # Standard CMake variable
            "$ENV{CONDA_PREFIX}/lib"           # Direct conda env check
        NO_DEFAULT_PATH
    )
    
    if(NOT NVRTC_LIBRARY)
        # Fallback to system paths
        find_library(NVRTC_LIBRARY NAMES nvrtc)
    endif()
    
    if(NVRTC_LIBRARY)
        message(STATUS "Found NVRTC: ${NVRTC_LIBRARY}")
        add_library(CUDA::nvrtc SHARED IMPORTED)
        set_target_properties(CUDA::nvrtc PROPERTIES
            IMPORTED_LOCATION "${NVRTC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
        )
    else()
        message(WARNING "NVRTC not found - JIT compilation will not work")
        message(STATUS "  Looked in: ${_cuda_prefix}/lib, ${CUDAToolkit_LIBRARY_DIR}")
    endif()
endif()

# Link against CUDA libraries
# - CUDA::cuda_driver: Low-level driver API (cuModuleLoad, cuLaunchKernel, etc.)
# - CUDA::nvrtc: Runtime compilation (nvrtcCompileProgram, etc.)
# - CUDA::cudart: Runtime API (cudaMalloc, etc.)
target_link_libraries(sinter_core
    CUDA::cuda_driver
    CUDA::cudart
)

# Link NVRTC if available
if(TARGET CUDA::nvrtc)
    target_link_libraries(sinter_core CUDA::nvrtc)
else()
    target_compile_definitions(sinter_core PRIVATE SINTER_NO_NVRTC)
endif()

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
    target_link_libraries(codegen_test PRIVATE sinter_core GTest::gtest_main CUDA::cudart)
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
    
    # LibTorch sanity test (only if LibTorch is available)
    if(SINTER_HAS_LIBTORCH)
        add_executable(libtorch_sanity_test test/integration/libtorch_sanity_test.cpp)
        target_link_libraries(libtorch_sanity_test ${TORCH_LIBRARIES})
        target_include_directories(libtorch_sanity_test PRIVATE ${TORCH_INCLUDE_DIRS})
        
        # LibTorch requires these compile definitions
        target_compile_definitions(libtorch_sanity_test PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
        
        # Add to CTest
        add_test(NAME LibTorchSanityTest COMMAND libtorch_sanity_test)
    endif()
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
