/**
 * @file ptx_compiler.cpp
 * @brief NVRTC-based runtime CUDA compilation
 *
 * This file implements runtime compilation of CUDA source code using NVRTC
 * (NVIDIA Runtime Compilation library).
 *
 * NVRTC Flow:
 *   1. Create program from source string
 *   2. Compile to PTX (parallel thread execution assembly)
 *   3. Load PTX into CUDA context as a module
 *   4. Extract the kernel function handle
 *
 * Key NVRTC functions:
 *   nvrtcCreateProgram  - Create compilation unit from source
 *   nvrtcCompileProgram - Compile to PTX
 *   nvrtcGetPTX         - Get PTX output
 *   nvrtcGetProgramLog  - Get error messages
 *
 * Key CUDA Driver API functions:
 *   cuInit              - Initialize driver
 *   cuModuleLoadData    - Load PTX as module
 *   cuModuleGetFunction - Get function handle
 *   cuLaunchKernel      - Execute kernel
 */

#include "sinter/codegen.h"
#include <nvrtc.h>
#include <cuda.h>
#include <stdexcept>
#include <iostream>

namespace sinter {

bool NvrtcCompiler::cuda_initialized_ = false;

#define NVRTC_CHECK(call)                                                                          \
    do {                                                                                           \
        nvrtcResult result = call;                                                                 \
        if (result != NVRTC_SUCCESS) {                                                             \
            throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(result) +  \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                                          \
    } while (0)

#define CUDA_DRIVER_CHECK(call)                                                                    \
    do {                                                                                           \
        CUresult result = call;                                                                    \
        if (result != CUDA_SUCCESS) {                                                              \
            const char* error_str;                                                                 \
            cuGetErrorString(result, &error_str);                                                  \
            throw std::runtime_error(std::string("CUDA Driver error: ") + error_str + " at " +     \
                                     __FILE__ + ":" + std::to_string(__LINE__));                   \
        }                                                                                          \
    } while (0)

void NvrtcCompiler::initialize_cuda() {
    if (cuda_initialized_) return;

    // Initialize CUDA Driver API
    CUDA_DRIVER_CHECK(cuInit(0));

    // Create a context on device 0
    // Note: In a real application, you'd want to manage contexts more carefully
    CUdevice device;
    CUDA_DRIVER_CHECK(cuDeviceGet(&device, 0));

    CUcontext context;
    CUDA_DRIVER_CHECK(cuCtxCreate(&context, 0, device));

    // Get device name for logging
    char device_name[256];
    CUDA_DRIVER_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    std::cout << "[Sinter] CUDA initialized on: " << device_name << '\n';

    cuda_initialized_ = true;
}

bool NvrtcCompiler::is_cuda_initialized() {
    return cuda_initialized_;
}

std::string NvrtcCompiler::compile_to_ptx(const std::string& cuda_source,
                                          const std::vector<std::string>& compile_options) {
    // Create NVRTC program
    nvrtcProgram program;
    NVRTC_CHECK(nvrtcCreateProgram(&program,
                                   cuda_source.c_str(), // Source code
                                   "sinter_kernel.cu",  // Name (for error messages)
                                   0,                   // Number of headers
                                   nullptr,             // Header contents
                                   nullptr              // Header names
                                   ));

    // Prepare compile options
    std::vector<const char*> options;
    options.reserve(compile_options.size() + 3); // + the 3 default flags below
    for (const auto& opt : compile_options) {
        options.push_back(opt.c_str());
    }

    // Always add some default options
    // -default-device makes functions __device__ by default
    // -std=c++17 for modern C++ features
    std::vector<std::string> default_opts = {
        "--gpu-architecture=compute_89", // RTX 4060 (Ada Lovelace)
        "-std=c++17", "-default-device"};
    for (const auto& opt : default_opts) {
        options.push_back(opt.c_str());
    }

    nvrtcResult compile_result =
        nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());

    // log size includes the null terminator, so 1 means empty
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));

    if (log_size > 1) {
        std::string log(log_size, '\0');
        NVRTC_CHECK(nvrtcGetProgramLog(program, &log[0]));

        if (compile_result != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&program);
            throw std::runtime_error("NVRTC compile failed:\n" + log);
        }
        std::cerr << "[Sinter] NVRTC warnings:\n" << log << '\n';
    } else if (compile_result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&program);
        throw std::runtime_error("NVRTC compilation failed with no log");
    }

    // Get PTX
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));

    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(program, &ptx[0]));

    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    return ptx;
}

CompiledKernel NvrtcCompiler::compile(const std::string& cuda_source,
                                      const std::string& kernel_name,
                                      const std::vector<std::string>& compile_options) {
    // Ensure CUDA is initialized
    if (!cuda_initialized_) {
        initialize_cuda();
    }

    // Compile to PTX
    std::string ptx = compile_to_ptx(cuda_source, compile_options);

    // Load PTX as a CUDA module
    CUmodule module;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&module, ptx.c_str()));

    // kernel_name has to match the extern "C" name, no mangling
    CUfunction function;
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&function, module, kernel_name.c_str()));

    CompiledKernel kernel;
    kernel.module_ = module;
    kernel.function_ = function;
    kernel.name_ = kernel_name;
    kernel.ptx_ = std::move(ptx);
    return kernel;
}

CompiledKernel::~CompiledKernel() {
    if (module_) {
        cuModuleUnload(module_);
        module_ = nullptr;
        function_ = nullptr;
    }
}

CompiledKernel::CompiledKernel(CompiledKernel&& other) noexcept
    : module_(other.module_), function_(other.function_), name_(std::move(other.name_)),
      ptx_(std::move(other.ptx_)) {
    other.module_ = nullptr;
    other.function_ = nullptr;
}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept {
    if (this != &other) {
        // Clean up our resources
        if (module_) {
            cuModuleUnload(module_);
        }

        // Move from other
        module_ = other.module_;
        function_ = other.function_;
        name_ = std::move(other.name_);
        ptx_ = std::move(other.ptx_);

        // Clear other
        other.module_ = nullptr;
        other.function_ = nullptr;
    }
    return *this;
}

void CompiledKernel::launch(unsigned int grid_dim, unsigned int block_dim, void** args,
                            CUstream stream) {
    if (!function_) {
        throw std::runtime_error("Cannot launch invalid kernel");
    }

    // Launch the kernel using CUDA Driver API
    // cuLaunchKernel is the low-level way to launch kernels
    CUDA_DRIVER_CHECK(cuLaunchKernel(function_, grid_dim, 1, 1, // Grid dimensions (x, y, z)
                                     block_dim, 1, 1,           // Block dimensions (x, y, z)
                                     0,                         // Shared memory bytes
                                     stream,                    // Stream
                                     args,                      // Kernel arguments
                                     nullptr                    // Extra (unused)
                                     ));
}

}
