#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "../include/MatrixFP32.cuh"
#include "../include/MatrixFP16.cuh"
#include "../include/utils.cuh"
#include "../include/naive_tensor_tgemm.cuh"

// Structure to hold benchmark results
struct BenchmarkResult {
    int matrix_size;
    int batch_num;
    double cublas_time_ms;
    double naive_time_ms;
    double cublas_gflops;
    double naive_gflops;
    double speedup;
    float max_error;
    float avg_error;
};

// Compute GFLOPS for matrix multiplication: 2*n^3 operations
double compute_gflops(int n, double time_seconds) {
    return (2.0 * n * n * n * 1e-9) / time_seconds;
}

// Compute statistics
void compute_error_stats(const MatrixFP32& expected, const MatrixFP32& actual, 
                         float& max_error, float& avg_error) {
    max_error = 0.0f;
    avg_error = 0.0f;
    
    for (int i = 0; i < expected.n_rows * expected.n_cols; i++) {
        float error = fabsf(expected.ptr[i] - actual.ptr[i]) / (fabsf(expected.ptr[i]) + 1e-6f);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= (expected.n_rows * expected.n_cols);
}

// Write results to CSV file
void write_results_csv(const std::vector<BenchmarkResult>& results, 
                       const std::string& filename) {
    std::ofstream file(filename);
    file << "MatrixSize,Batch,cuBLAS_Time_ms,Naive_Time_ms,cuBLAS_GFLOPS,Naive_GFLOPS,"
         << "Speedup,MaxError,AvgError\n";
    
    for (const auto& r : results) {
        file << r.matrix_size << "," << r.batch_num << ","
             << std::fixed << std::setprecision(6)
             << r.cublas_time_ms << "," << r.naive_time_ms << ","
             << r.cublas_gflops << "," << r.naive_gflops << ","
             << r.speedup << "," << r.max_error << "," << r.avg_error << "\n";
    }
    file.close();
}

int main(int argc, char const *argv[])
{
    std::cout << "========================================\n";
    std::cout << "  Mixed Precision GEMM Benchmark Test   \n";
    std::cout << "========================================\n\n";
    
    // Extended matrix sizes for comprehensive testing
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    int num_batches = 3;
    int runs_per_batch = 5;
    
    std::vector<BenchmarkResult> all_results;
    
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n\n";
    
    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    // Main benchmark loop
    for (int batch = 0; batch < num_batches; batch++) {
        std::cout << "=== Batch " << (batch + 1) << " / " << num_batches << " ===\n";
        
        for (int mat_idx = 0; mat_idx < n_sizes; mat_idx++) {
            int n = mat_sizes[mat_idx];
            std::cout << "  Matrix Size: " << n << "x" << n << " ... ";
            std::cout.flush();
            
            // Check memory availability
            size_t required_memory = 3 * n * n * sizeof(float) + 2 * n * n * sizeof(__half);
            cudaMemGetInfo(nullptr, nullptr); // Reset memory tracking
            
            // Define matrices
            MatrixFP16 A_FP16 = MatrixFP16(n, n, false);
            MatrixFP16 B_FP16 = MatrixFP16(n, n, false);
            MatrixFP32 C_FP32_cublas = MatrixFP32(n, n, false);
            MatrixFP32 C_FP32_naive = MatrixFP32(n, n, false);
            MatrixFP32 C_FP32_ref = MatrixFP32(n, n, false);
            
            // Initialize matrices with random values
            random_init_mat(A_FP16, -1.0f, 1.0f);
            random_init_mat(B_FP16, -1.0f, 1.0f);
            init_mat(C_FP32_cublas, 0.0f);
            init_mat(C_FP32_naive, 0.0f);
            init_mat(C_FP32_ref, 0.0f);
            
            // Move to device
            MatrixFP16 d_A_FP16 = MatrixFP16(n, n, true);
            MatrixFP16 d_B_FP16 = MatrixFP16(n, n, true);
            MatrixFP32 d_C_FP32_cublas = MatrixFP32(n, n, true);
            MatrixFP32 d_C_FP32_naive = MatrixFP32(n, n, true);
            MatrixFP32 d_C_FP32_ref = MatrixFP32(n, n, true);
            
            A_FP16.copy_to_device(d_A_FP16);
            B_FP16.copy_to_device(d_B_FP16);
            C_FP32_cublas.copy_to_device(d_C_FP32_cublas);
            C_FP32_naive.copy_to_device(d_C_FP32_naive);
            C_FP32_ref.copy_to_device(d_C_FP32_ref);
            cudaDeviceSynchronize();
            
            // ---- Warmup Runs ----
            cublasHandle_t handle;
            cublas_check(cublasCreate(&handle));
            
            for (int w = 0; w < 2; w++) {
                float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    d_B_FP16.ptr, CUDA_R_16F, n,
                    d_A_FP16.ptr, CUDA_R_16F, n,
                    &beta,
                    d_C_FP32_ref.ptr, CUDA_R_32F, n,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                cudaDeviceSynchronize();
            }
            
            for (int w = 0; w < 2; w++) {
                naive_tensor_tgemm(d_A_FP16.ptr, d_B_FP16.ptr, d_C_FP32_naive.ptr, n, n, n);
                cudaDeviceSynchronize();
            }
            
            // ---- cuBLAS Benchmark ----
            double cublas_time_total = 0.0;
            
            // Reset C matrix
            init_mat(C_FP32_cublas, 0.0f);
            C_FP32_cublas.copy_to_device(d_C_FP32_cublas);
            cudaDeviceSynchronize();
            
            cudaEventRecord(beg);
            for (int r = 0; r < runs_per_batch; r++) {
                float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n,
                    &alpha,
                    d_B_FP16.ptr, CUDA_R_16F, n,
                    d_A_FP16.ptr, CUDA_R_16F, n,
                    &beta,
                    d_C_FP32_cublas.ptr, CUDA_R_32F, n,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                cudaDeviceSynchronize();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, beg, end);
            cublas_time_total = elapsed_time / runs_per_batch;
            
            // ---- Naive Kernel Benchmark ----
            double naive_time_total = 0.0;
            
            // Reset C matrix
            init_mat(C_FP32_naive, 0.0f);
            C_FP32_naive.copy_to_device(d_C_FP32_naive);
            cudaDeviceSynchronize();
            
            cudaEventRecord(beg);
            for (int r = 0; r < runs_per_batch; r++) {
                naive_tensor_tgemm(d_A_FP16.ptr, d_B_FP16.ptr, d_C_FP32_naive.ptr, n, n, n);
                cudaDeviceSynchronize();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, beg, end);
            naive_time_total = elapsed_time / runs_per_batch;
            
            // ---- Copy results back and validate ----
            d_C_FP32_cublas.copy_to_host(C_FP32_cublas);
            d_C_FP32_naive.copy_to_host(C_FP32_naive);
            d_C_FP32_ref.copy_to_host(C_FP32_ref);
            cudaDeviceSynchronize();
            
            // Compute error statistics
            float max_error = 0.0f, avg_error = 0.0f;
            compute_error_stats(C_FP32_ref, C_FP32_naive, max_error, avg_error);
            
            // Convert to seconds
            double cublas_time_sec = cublas_time_total / 1000.0;
            double naive_time_sec = naive_time_total / 1000.0;
            
            // Calculate GFLOPS
            double cublas_gflops = compute_gflops(n, cublas_time_sec);
            double naive_gflops = compute_gflops(n, naive_time_sec);
            double speedup = cublas_time_sec / naive_time_sec;
            
            // Store results
            BenchmarkResult result;
            result.matrix_size = n;
            result.batch_num = batch + 1;
            result.cublas_time_ms = cublas_time_total;
            result.naive_time_ms = naive_time_total;
            result.cublas_gflops = cublas_gflops;
            result.naive_gflops = naive_gflops;
            result.speedup = speedup;
            result.max_error = max_error;
            result.avg_error = avg_error;
            
            all_results.push_back(result);
            
            std::cout << "cuBLAS: " << std::fixed << std::setprecision(3) 
                     << cublas_gflops << " GFLOPS | "
                     << "Naive: " << naive_gflops << " GFLOPS | "
                     << "Speedup: " << speedup << "x\n";
            
            // Cleanup
            cublas_check(cublasDestroy(handle));
            A_FP16.free_mat();
            B_FP16.free_mat();
            C_FP32_cublas.free_mat();
            C_FP32_naive.free_mat();
            C_FP32_ref.free_mat();
        }
        std::cout << "\n";
    }
    
    // Print summary statistics
    std::cout << "========================================\n";
    std::cout << "           Summary Statistics           \n";
    std::cout << "========================================\n\n";
    
    for (int mat_idx = 0; mat_idx < n_sizes; mat_idx++) {
        int n = mat_sizes[mat_idx];
        double avg_cublas_gflops = 0.0, avg_naive_gflops = 0.0;
        double avg_speedup = 0.0;
        int count = 0;
        
        for (const auto& r : all_results) {
            if (r.matrix_size == n) {
                avg_cublas_gflops += r.cublas_gflops;
                avg_naive_gflops += r.naive_gflops;
                avg_speedup += r.speedup;
                count++;
            }
        }
        
        avg_cublas_gflops /= count;
        avg_naive_gflops /= count;
        avg_speedup /= count;
        
        std::cout << "Size " << n << "x" << n << ": "
                 << "cuBLAS: " << std::fixed << std::setprecision(2) 
                 << avg_cublas_gflops << " GFLOPS | "
                 << "Naive: " << avg_naive_gflops << " GFLOPS | "
                 << "Avg Speedup: " << avg_speedup << "x\n";
    }
    
    // Write results to CSV
    write_results_csv(all_results, "benchmark_results.csv");
    std::cout << "\nResults saved to: benchmark_results.csv\n";
    
    // Cleanup
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    
    return 0;
}
