/**
 * @file codegen.h
 * @brief CUDA code generation and runtime compilation
 *
 * This module converts Sinter IR operations into executable CUDA kernels.
 *
 * The pipeline is:
 *   Sinter IR (Op)
 *       → CudaKernelGen::generate()
 *       → CUDA C++ source string
 *       → NvrtcCompiler::compile()
 *       → PTX
 *       → CUmodule/CUfunction (executable on GPU)
 *
 * Example usage:
 *   // 1. Generate CUDA source for an Add operation
 *   std::string cuda_src = CudaKernelGen::generate_add(shape);
 *
 *   // 2. Compile to PTX, then load as executable
 *   CompiledKernel kernel = NvrtcCompiler::compile(cuda_src, "add_kernel");
 *
 *   // 3. Launch the kernel
 *   kernel.launch(grid_dim, block_dim, args...);
 */

#pragma once

#include "sinter/ir.h"
#include <string>
#include <vector>
#include <memory>
#include <cuda.h>

namespace sinter {

/**
 * @class CompiledKernel
 * @brief Holds a compiled CUDA kernel ready for execution
 *
 * This wraps the CUDA Driver API objects (CUmodule, CUfunction) and provides
 * a simpler interface for launching kernels.
 */
class CompiledKernel {
public:
    CompiledKernel() = default;
    ~CompiledKernel();

    // Move-only (can't copy CUDA handles)
    CompiledKernel(CompiledKernel&& other) noexcept;
    CompiledKernel& operator=(CompiledKernel&& other) noexcept;
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    /**
     * Launch the kernel
     * @param grid_dim Number of blocks (gridDim.x)
     * @param block_dim Threads per block (blockDim.x)
     * @param args Kernel arguments as void* array
     * @param stream CUDA stream (0 for default)
     */
    void launch(unsigned int grid_dim, unsigned int block_dim, void** args,
                CUstream stream = nullptr);

    /**
     * Check if kernel is valid and ready to launch
     */
    bool is_valid() const { return function_ != nullptr; }

    /**
     * Get the kernel name
     */
    const std::string& name() const { return name_; }

private:
    friend class NvrtcCompiler;

    CUmodule module_ = nullptr;     // Loaded module (contains the kernel)
    CUfunction function_ = nullptr; // The kernel function handle
    std::string name_;              // Kernel name (for debugging)
    std::string ptx_;               // PTX source (for debugging)
};

/**
 * @class NvrtcCompiler
 * @brief Compiles CUDA C++ source to PTX using NVRTC, then loads it
 *
 * NVRTC (NVIDIA Runtime Compilation) lets us compile CUDA code at runtime.
 * This is essential for a JIT compiler like Sinter.
 *
 * The compilation flow:
 *   CUDA C++ string → NVRTC → PTX string → CUDA Driver → CUmodule
 */
class NvrtcCompiler {
public:
    /**
     * Compile CUDA source code to an executable kernel
     *
     * @param cuda_source The CUDA C++ source code
     * @param kernel_name Name of the __global__ function to extract
     * @param compile_options Additional NVCC options (e.g., "-arch=sm_89")
     * @return CompiledKernel ready for launch
     * @throws std::runtime_error if compilation fails
     */
    static CompiledKernel compile(const std::string& cuda_source, const std::string& kernel_name,
                                  const std::vector<std::string>& compile_options = {});

    /**
     * Compile to PTX only (for debugging/inspection)
     *
     * @param cuda_source The CUDA C++ source code
     * @param compile_options Additional NVCC options
     * @return PTX assembly as string
     */
    static std::string compile_to_ptx(const std::string& cuda_source,
                                      const std::vector<std::string>& compile_options = {});

    /**
     * Initialize the CUDA Driver API (call once at startup)
     * This is required before any kernel compilation/launch.
     */
    static void initialize_cuda();

    /**
     * Check if CUDA has been initialized
     */
    static bool is_cuda_initialized();

private:
    static bool cuda_initialized_;
};

/**
 * @class CudaKernelGen
 * @brief Generates CUDA C++ source code from Sinter IR
 *
 * Each method generates a complete, compilable CUDA kernel for a specific
 * operation. The generated code follows a consistent pattern:
 *
 *   extern "C" __global__ void kernel_name(args...) {
 *       int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *       if (idx < n) {
 *           // operation
 *       }
 *   }
 *
 * We use extern "C" to avoid C++ name mangling, making it easier to
 * look up the kernel by name.
 */
class CudaKernelGen {
public:
    /**
     * Generate element-wise Add kernel: C = A + B
     *
     * @param num_elements Total number of elements
     * @return CUDA source code string
     *
     * Generated kernel signature:
     *   add_kernel(const float* A, const float* B, float* C, int n)
     */
    static std::string generate_add(size_t num_elements);

    /**
     * Generate element-wise ReLU kernel: Y = max(0, X)
     *
     * @param num_elements Total number of elements
     * @return CUDA source code string
     *
     * Generated kernel signature:
     *   relu_kernel(const float* X, float* Y, int n)
     */
    static std::string generate_relu(size_t num_elements);

    /**
     * Generate fused Add+ReLU kernel: Y = max(0, A + B)
     *
     * @param num_elements Total number of elements
     * @return CUDA source code string
     *
     * This is what FusionPass optimizes Add→ReLU sequences into.
     * By fusing, we save one memory round-trip.
     *
     * Generated kernel signature:
     *   fused_add_relu_kernel(const float* A, const float* B, float* Y, int n)
     */
    static std::string generate_fused_add_relu(size_t num_elements);

    /**
     * Generate kernel from Sinter Op
     *
     * @param op The operation to generate code for
     * @param graph The graph (needed to look up input shapes)
     * @return CUDA source code string, or empty string if op not supported
     */
    static std::string generate_from_op(const Op& op, const Graph& graph);

    /**
     * Get the kernel function name for an operation type
     */
    static std::string get_kernel_name(OpType type);
};

/**
 * @class KernelLauncher
 * @brief Helper to compute launch configuration and execute kernels
 *
 * Determines optimal grid/block dimensions based on:
 * - Number of elements to process
 * - GPU properties (max threads per block, etc.)
 */
class KernelLauncher {
public:
    /**
     * Compute launch configuration for element-wise operations
     *
     * @param num_elements Total elements to process
     * @param threads_per_block Desired threads per block (default 256)
     * @return pair of (grid_dim, block_dim)
     */
    static std::pair<unsigned int, unsigned int>
    compute_launch_config(size_t num_elements, unsigned int threads_per_block = 256);
};

}
