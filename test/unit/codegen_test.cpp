/**
 * @file codegen_test.cpp
 * @brief Unit tests for CUDA code generation and compilation
 */

#include <gtest/gtest.h>
#include "sinter/codegen.h"
#include "sinter/ir.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace sinter;

// =============================================================================
// Test: Code Generation (no GPU required)
// =============================================================================

TEST(CodegenTest, GenerateAddKernel) {
    std::string code = CudaKernelGen::generate_add(1024);
    
    // Check that the generated code contains expected elements
    EXPECT_NE(code.find("add_kernel"), std::string::npos);
    EXPECT_NE(code.find("__global__"), std::string::npos);
    EXPECT_NE(code.find("extern \"C\""), std::string::npos);
    EXPECT_NE(code.find("A[idx] + B[idx]"), std::string::npos);
}

TEST(CodegenTest, GenerateReluKernel) {
    std::string code = CudaKernelGen::generate_relu(1024);
    
    EXPECT_NE(code.find("relu_kernel"), std::string::npos);
    EXPECT_NE(code.find("__global__"), std::string::npos);
}

TEST(CodegenTest, GenerateFusedAddReluKernel) {
    std::string code = CudaKernelGen::generate_fused_add_relu(1024);
    
    EXPECT_NE(code.find("fused_add_relu_kernel"), std::string::npos);
    EXPECT_NE(code.find("__global__"), std::string::npos);
    // Should have both add and relu logic
    EXPECT_NE(code.find("A[idx] + B[idx]"), std::string::npos);
}

TEST(CodegenTest, GenerateFromOp) {
    Graph graph;
    graph.add_op(Op(0, OpType::Input, {}, Shape({1024})));
    graph.add_op(Op(1, OpType::Input, {}, Shape({1024})));
    graph.add_op(Op(2, OpType::Add, {0, 1}, Shape({1024})));
    
    std::string code = CudaKernelGen::generate_from_op(graph.get_op(2), graph);
    EXPECT_NE(code.find("add_kernel"), std::string::npos);
}

TEST(CodegenTest, KernelNameMapping) {
    EXPECT_EQ(CudaKernelGen::get_kernel_name(OpType::Add), "add_kernel");
    EXPECT_EQ(CudaKernelGen::get_kernel_name(OpType::ReLU), "relu_kernel");
    EXPECT_EQ(CudaKernelGen::get_kernel_name(OpType::MatMul), "matmul_kernel");
}

TEST(CodegenTest, LaunchConfig) {
    // 1024 elements, 256 threads per block -> 4 blocks
    auto [grid, block] = KernelLauncher::compute_launch_config(1024, 256);
    EXPECT_EQ(grid, 4u);
    EXPECT_EQ(block, 256u);
    
    // 1000 elements, 256 threads -> ceil(1000/256) = 4 blocks
    auto [grid2, block2] = KernelLauncher::compute_launch_config(1000, 256);
    EXPECT_EQ(grid2, 4u);
    EXPECT_EQ(block2, 256u);
    
    // Edge case: 0 elements -> at least 1 block
    auto [grid3, block3] = KernelLauncher::compute_launch_config(0, 256);
    EXPECT_EQ(grid3, 1u);
}

// =============================================================================
// Test: NVRTC Compilation (requires CUDA)
// =============================================================================

class NvrtcTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(NvrtcTest, CompilesToPtx) {
    std::string code = CudaKernelGen::generate_add(1024);
    std::string ptx = NvrtcCompiler::compile_to_ptx(code);
    
    // PTX should contain some recognizable elements
    EXPECT_NE(ptx.find(".version"), std::string::npos);
    EXPECT_NE(ptx.find(".target"), std::string::npos);
    EXPECT_NE(ptx.find("add_kernel"), std::string::npos);
}

TEST_F(NvrtcTest, CompilesAndLoads) {
    std::string code = CudaKernelGen::generate_add(1024);
    
    CompiledKernel kernel = NvrtcCompiler::compile(code, "add_kernel");
    
    EXPECT_TRUE(kernel.is_valid());
    EXPECT_EQ(kernel.name(), "add_kernel");
}

// =============================================================================
// Test: End-to-End Kernel Execution (requires CUDA)
// =============================================================================

TEST_F(NvrtcTest, ExecuteAddKernel) {
    const int N = 1024;
    
    // 1. Generate and compile kernel
    std::string code = CudaKernelGen::generate_add(N);
    CompiledKernel kernel = NvrtcCompiler::compile(code, "add_kernel");
    ASSERT_TRUE(kernel.is_valid());
    
    // 2. Prepare test data on host
    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // 3. Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // 4. Copy input to GPU
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 5. Launch kernel
    auto [grid_dim, block_dim] = KernelLauncher::compute_launch_config(N);
    int n = N;  // Kernel expects int
    void* args[] = { &d_A, &d_B, &d_C, &n };
    kernel.launch(grid_dim, block_dim, args);
    
    // 6. Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);
    
    // 7. Copy result back
    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 8. Verify results
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        EXPECT_FLOAT_EQ(h_C[i], expected) << "Mismatch at index " << i;
    }
    
    // 9. Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(NvrtcTest, ExecuteReluKernel) {
    const int N = 1024;
    
    // Generate and compile
    std::string code = CudaKernelGen::generate_relu(N);
    CompiledKernel kernel = NvrtcCompiler::compile(code, "relu_kernel");
    ASSERT_TRUE(kernel.is_valid());
    
    // Prepare data (half negative, half positive)
    std::vector<float> h_X(N), h_Y(N);
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i - N/2);  // -512 to +511
    }
    
    // Allocate and copy
    float *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));
    cudaMemcpy(d_X, h_X.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch
    auto [grid_dim, block_dim] = KernelLauncher::compute_launch_config(N);
    int n = N;
    void* args[] = { &d_X, &d_Y, &n };
    kernel.launch(grid_dim, block_dim, args);
    cudaDeviceSynchronize();
    
    // Copy back and verify
    cudaMemcpy(h_Y.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i) {
        float expected = std::max(0.0f, h_X[i]);
        EXPECT_FLOAT_EQ(h_Y[i], expected) << "Mismatch at index " << i;
    }
    
    cudaFree(d_X);
    cudaFree(d_Y);
}

TEST_F(NvrtcTest, ExecuteFusedAddReluKernel) {
    const int N = 1024;
    
    // Generate and compile
    std::string code = CudaKernelGen::generate_fused_add_relu(N);
    CompiledKernel kernel = NvrtcCompiler::compile(code, "fused_add_relu_kernel");
    ASSERT_TRUE(kernel.is_valid());
    
    // Prepare data
    std::vector<float> h_A(N), h_B(N), h_Y(N);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i - N/2);  // -512 to +511
        h_B[i] = -100.0f;  // Make some sums negative
    }
    
    // Allocate and copy
    float *d_A, *d_B, *d_Y;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch
    auto [grid_dim, block_dim] = KernelLauncher::compute_launch_config(N);
    int n = N;
    void* args[] = { &d_A, &d_B, &d_Y, &n };
    kernel.launch(grid_dim, block_dim, args);
    cudaDeviceSynchronize();
    
    // Copy back and verify
    cudaMemcpy(h_Y.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; ++i) {
        float sum = h_A[i] + h_B[i];
        float expected = std::max(0.0f, sum);
        EXPECT_FLOAT_EQ(h_Y[i], expected) << "Mismatch at index " << i;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Y);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
