/**
 * @file cuda_kernel_gen.cpp
 * @brief Implementation of CUDA kernel code generation
 * 
 * This file generates CUDA C++ source code strings for various operations.
 * The generated code follows a consistent pattern optimized for:
 * - Simplicity (easy to understand and debug)
 * - Correctness (bounds checking, proper indexing)
 * 
 * Future optimizations (Phase 3+):
 * - Tiled memory access for better cache usage
 * - Shared memory utilization
 * - Vectorized loads (float4)
 * - Warp-level primitives
 */

#include "sinter/codegen.h"
#include <sstream>

namespace sinter {

std::string CudaKernelGen::generate_add(size_t num_elements) {
    // We embed the element count in a comment for debugging,
    // but the actual count is passed as a kernel parameter 'n'
    // so the same compiled kernel can work for different sizes.
    
    std::ostringstream oss;
    
    oss << R"(
// =============================================================================
// Sinter Generated Kernel: Element-wise Add
// Operation: C[i] = A[i] + B[i]
// Elements: )" << num_elements << R"(
// =============================================================================

extern "C" __global__ void add_kernel(
    const float* __restrict__ A,    // Input tensor A
    const float* __restrict__ B,    // Input tensor B
    float* __restrict__ C,          // Output tensor C = A + B
    int n                           // Number of elements
) {
    // Calculate global thread index
    // Each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check: don't access beyond array
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
)";
    
    return oss.str();
}

std::string CudaKernelGen::generate_relu(size_t num_elements) {
    std::ostringstream oss;
    
    oss << R"(
// =============================================================================
// Sinter Generated Kernel: ReLU Activation
// Operation: Y[i] = max(0, X[i])
// Elements: )" << num_elements << R"(
// =============================================================================

extern "C" __global__ void relu_kernel(
    const float* __restrict__ X,    // Input tensor
    float* __restrict__ Y,          // Output tensor Y = ReLU(X)
    int n                           // Number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = X[idx];
        // ReLU: max(0, x)
        // Using ternary instead of fmaxf for clarity
        // (compiler will optimize this)
        Y[idx] = (val > 0.0f) ? val : 0.0f;
    }
}
)";
    
    return oss.str();
}

std::string CudaKernelGen::generate_fused_add_relu(size_t num_elements) {
    std::ostringstream oss;
    
    oss << R"(
// =============================================================================
// Sinter Generated Kernel: Fused Add + ReLU
// Operation: Y[i] = max(0, A[i] + B[i])
// Elements: )" << num_elements << R"(
// 
// This kernel fuses two operations into one, saving memory bandwidth:
// - Without fusion: Add writes to memory, ReLU reads from memory
// - With fusion: Intermediate result stays in registers
// =============================================================================

extern "C" __global__ void fused_add_relu_kernel(
    const float* __restrict__ A,    // Input tensor A
    const float* __restrict__ B,    // Input tensor B
    float* __restrict__ Y,          // Output tensor Y = ReLU(A + B)
    int n                           // Number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Compute A + B in register (no memory write for intermediate)
        float sum = A[idx] + B[idx];
        
        // Apply ReLU
        Y[idx] = (sum > 0.0f) ? sum : 0.0f;
    }
}
)";
    
    return oss.str();
}

std::string CudaKernelGen::generate_from_op(const Op& op, const Graph& graph) {
    // graph parameter reserved for future use (e.g., looking up input shapes)
    (void)graph;
    
    // Get the number of elements from output shape
    size_t num_elements = op.output_shape.numel();
    
    switch (op.type) {
        case OpType::Add:
            return generate_add(num_elements);
            
        case OpType::ReLU:
            return generate_relu(num_elements);
            
        case OpType::FusedOp: {
            // Check what operations are fused
            std::string fused_ops = op.get_attribute("fused_ops");
            if (fused_ops == "add,relu") {
                return generate_fused_add_relu(num_elements);
            }
            // Unknown fusion - return empty
            return "";
        }
        
        // TODO: Implement more operations
        case OpType::MatMul:
        case OpType::Multiply:
        case OpType::Softmax:
        case OpType::LayerNorm:
        case OpType::GELU:
        case OpType::FlashAttention:
        case OpType::Reshape:
        case OpType::Transpose:
        case OpType::Input:
        case OpType::Output:
        default:
            // Not yet implemented
            return "";
    }
}

std::string CudaKernelGen::get_kernel_name(OpType type) {
    switch (type) {
        case OpType::Add:       return "add_kernel";
        case OpType::ReLU:      return "relu_kernel";
        case OpType::FusedOp:   return "fused_add_relu_kernel";  // Assumes add+relu fusion
        case OpType::MatMul:    return "matmul_kernel";
        case OpType::Multiply:  return "multiply_kernel";
        case OpType::Softmax:   return "softmax_kernel";
        case OpType::LayerNorm: return "layernorm_kernel";
        case OpType::GELU:      return "gelu_kernel";
        default:                return "unknown_kernel";
    }
}

// =============================================================================
// KernelLauncher implementation
// =============================================================================

std::pair<unsigned int, unsigned int> 
KernelLauncher::compute_launch_config(size_t num_elements, unsigned int threads_per_block) {
    // Simple 1D launch configuration
    // grid_dim = ceil(num_elements / threads_per_block)
    unsigned int grid_dim = (static_cast<unsigned int>(num_elements) + threads_per_block - 1) 
                          / threads_per_block;
    
    // Ensure at least 1 block
    if (grid_dim == 0) grid_dim = 1;
    
    return {grid_dim, threads_per_block};
}

} // namespace sinter
