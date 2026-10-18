/**
 * @file libtorch_sanity_test.cpp
 * @brief Sanity check that LibTorch is correctly linked and working
 * 
 * This test verifies:
 * 1. We can include torch headers
 * 2. We can create tensors
 * 3. We can perform basic operations (matmul)
 * 4. CUDA tensors work (if GPU available)
 */

#include <torch/torch.h>
#include <iostream>
#include <chrono>

void test_cpu_matmul() {
    std::cout << "=== CPU MatMul Test ===" << std::endl;
    
    // Create two random matrices
    auto A = torch::rand({1024, 1024});
    auto B = torch::rand({1024, 1024});
    
    std::cout << "A shape: " << A.sizes() << std::endl;
    std::cout << "B shape: " << B.sizes() << std::endl;
    
    // Warm-up
    auto C = torch::matmul(A, B);
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        C = torch::matmul(A, B);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "C shape: " << C.sizes() << std::endl;
    std::cout << "10 iterations took: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Average per matmul: " << duration.count() / 10000.0 << " ms" << std::endl;
    std::cout << "CPU test PASSED" << std::endl;
}

void test_cuda_matmul() {
    std::cout << "\n=== CUDA MatMul Test ===" << std::endl;
    
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA not available, skipping GPU test" << std::endl;
        return;
    }
    
    std::cout << "CUDA is available!" << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    // Create tensors on GPU
    auto A = torch::rand({1024, 1024}, torch::device(torch::kCUDA));
    auto B = torch::rand({1024, 1024}, torch::device(torch::kCUDA));
    
    std::cout << "A device: " << A.device() << std::endl;
    std::cout << "B device: " << B.device() << std::endl;
    
    // Warm-up (important for GPU!)
    auto C = torch::matmul(A, B);
    torch::cuda::synchronize();  // Wait for GPU
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        C = torch::matmul(A, B);
    }
    torch::cuda::synchronize();  // Wait for all operations to complete
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "C shape: " << C.sizes() << std::endl;
    std::cout << "100 iterations took: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Average per matmul: " << duration.count() / 100000.0 << " ms" << std::endl;
    
    // Calculate TFLOPS
    // MatMul of [M,K] x [K,N] = 2*M*N*K FLOPs
    double flops_per_matmul = 2.0 * 1024 * 1024 * 1024;
    double seconds_per_matmul = (duration.count() / 100.0) / 1e6;
    double tflops = (flops_per_matmul / seconds_per_matmul) / 1e12;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    
    std::cout << "CUDA test PASSED" << std::endl;
}

void test_add_relu() {
    std::cout << "\n=== Add + ReLU Test (Fusion Candidate) ===" << std::endl;
    
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << std::endl;
    
    auto A = torch::randn({1024, 1024}, torch::device(device));
    auto B = torch::randn({1024, 1024}, torch::device(device));
    
    // Unfused: Add then ReLU (two separate operations)
    if (torch::cuda::is_available()) torch::cuda::synchronize();
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto sum = torch::add(A, B);
        auto result = torch::relu(sum);
        (void)result;  // Prevent optimization
    }
    if (torch::cuda::is_available()) torch::cuda::synchronize();
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "Unfused (Add + ReLU separate): " << duration1.count() / 100.0 << " us/iter" << std::endl;
    
    // Note: PyTorch doesn't have a built-in fused add+relu, but this shows the baseline
    // Sinter's fusion pass should eventually beat the unfused version
    
    std::cout << "Add+ReLU test PASSED" << std::endl;
}

int main() {
    std::cout << "LibTorch Sanity Test" << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "===================" << std::endl;
    
    try {
        test_cpu_matmul();
        test_cuda_matmul();
        test_add_relu();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
