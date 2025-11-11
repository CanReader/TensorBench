#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "../include/MatrixFP32.cuh"
#include "../include/MatrixFP16.cuh"
#include "../include/utils.cuh"
#include "../include/naive_tensor_tgemm.cuh"

// Structure to hold stress test results
struct StressTestResult {
    int matrix_size;
    int num_runs;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_deviation;
    double gflops;
    double memory_allocated_gb;
    bool passed_validation;
    float max_numerical_error;
};

// Compute GFLOPS
double compute_gflops(int n, double time_seconds) {
    return (2.0 * n * n * n * 1e-9) / time_seconds;
}

// Compute standard deviation
double compute_stddev(const std::vector<double>& times, double mean) {
    double sum_sq_diff = 0.0;
    for (double t : times) {
        sum_sq_diff += (t - mean) * (t - mean);
    }
    return std::sqrt(sum_sq_diff / times.size());
}

// Validate numerical accuracy
float validate_accuracy(const MatrixFP32& reference, const MatrixFP32& result) {
    float max_error = 0.0f;
    for (int i = 0; i < reference.n_rows * reference.n_cols; i++) {
        float ref_val = reference.ptr[i];
        float res_val = result.ptr[i];
        float relative_error = std::fabs(ref_val - res_val) / (std::fabs(ref_val) + 1e-8f);
        max_error = std::max(max_error, relative_error);
    }
    return max_error;
}

int main(int argc, char const *argv[])
{
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║  TensorBench - Stress Test & Memory Hierarchy║\n";
    std::cout << "║  Peak Performance & Stability Analysis      ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    // Stress test parameters: Larger matrices to test memory limits
    int mat_sizes[] = {512, 1024, 2048, 4096, 5120, 6144};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    int stress_runs = 50; // Many runs to test stability

    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));

    std::vector<StressTestResult> results;

    std::cout << "Starting stress test with " << stress_runs << " iterations per size...\n\n";

    for (int mat_idx = 0; mat_idx < n_sizes; mat_idx++) {
        int n = mat_sizes[mat_idx];
        std::cout << "Testing Matrix Size: " << n << " x " << n;
        std::cout << " (Memory per matrix: " << (2.0 * n * n / (1024*1024)) << " MB)\n";
        std::cout << std::string(70, '-') << "\n";

        try {
            // Allocate host matrices
            MatrixFP16 h_A = MatrixFP16(n, n, false);
            MatrixFP16 h_B = MatrixFP16(n, n, false);
            MatrixFP32 h_C_cublas = MatrixFP32(n, n, false);
            MatrixFP32 h_C_naive = MatrixFP32(n, n, false);

            // Initialize with small random values to avoid overflow
            random_init_mat(h_A, -2, 2);
            random_init_mat(h_B, -2, 2);
            init_mat(h_C_cublas, 0.0f);
            init_mat(h_C_naive, 0.0f);

            // Allocate device matrices
            MatrixFP16 d_A = MatrixFP16(n, n, true);
            MatrixFP16 d_B = MatrixFP16(n, n, true);
            MatrixFP32 d_C_cublas = MatrixFP32(n, n, true);
            MatrixFP32 d_C_naive = MatrixFP32(n, n, true);

            // Copy to device
            h_A.copy_to_device(d_A);
            h_B.copy_to_device(d_B);
            h_C_cublas.copy_to_device(d_C_cublas);
            h_C_naive.copy_to_device(d_C_naive);
            cudaDeviceSynchronize();

            // Calculate memory allocated
            double memory_gb = 4.0 * (2.0 * n * n + 4.0 * n * n) / (1024*1024*1024); // A, B in FP16; C in FP32

            // Warmup runs
            for (int i = 0; i < 5; i++) {
                float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasGemmEx(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, n, n,
                                        &alpha,
                                        d_B.ptr, CUDA_R_16F, n,
                                        d_A.ptr, CUDA_R_16F, n,
                                        &beta,
                                        d_C_cublas.ptr, CUDA_R_32F, n,
                                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            cudaDeviceSynchronize();

            // Run stress test with multiple iterations
            std::vector<double> execution_times;
            std::vector<double> gflops_values;

            std::cout << "  Running " << stress_runs << " iterations";
            for (int run = 0; run < stress_runs; run++) {
                if (run % 10 == 0) std::cout << ".";
                std::cout.flush();

                // Reset result matrices
                init_mat(h_C_cublas, 0.0f);
                init_mat(h_C_naive, 0.0f);
                h_C_cublas.copy_to_device(d_C_cublas);
                h_C_naive.copy_to_device(d_C_naive);

                // Time a single cuBLAS execution
                cudaEventRecord(beg);
                float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasGemmEx(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, n, n,
                                        &alpha,
                                        d_B.ptr, CUDA_R_16F, n,
                                        d_A.ptr, CUDA_R_16F, n,
                                        &beta,
                                        d_C_cublas.ptr, CUDA_R_32F, n,
                                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                cudaEventRecord(end);
                cudaEventSynchronize(beg);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&elapsed_time, beg, end);

                double time_seconds = elapsed_time / 1000.0;
                double gflops = compute_gflops(n, time_seconds);
                execution_times.push_back(time_seconds);
                gflops_values.push_back(gflops);

                // Also run naive kernel for comparison on first run
                if (run == 0) {
                    naive_tensor_tgemm(d_A.ptr, d_B.ptr, d_C_naive.ptr, n, n, n);
                    cudaDeviceSynchronize();
                }
            }
            std::cout << "\n";

            // Compute statistics
            double min_time = *std::min_element(execution_times.begin(), execution_times.end());
            double max_time = *std::max_element(execution_times.begin(), execution_times.end());
            double avg_time = 0.0;
            for (double t : execution_times) avg_time += t;
            avg_time /= execution_times.size();
            double stddev = compute_stddev(execution_times, avg_time);

            double min_gflops = *std::min_element(gflops_values.begin(), gflops_values.end());
            double max_gflops = *std::max_element(gflops_values.begin(), gflops_values.end());
            double avg_gflops = 0.0;
            for (double g : gflops_values) avg_gflops += g;
            avg_gflops /= gflops_values.size();

            // Validate accuracy
            d_C_cublas.copy_to_host(h_C_cublas);
            d_C_naive.copy_to_host(h_C_naive);
            float max_error = validate_accuracy(h_C_cublas, h_C_naive);
            bool passed = max_error < 1e-3; // Allow 0.1% error for stress testing

            StressTestResult res;
            res.matrix_size = n;
            res.num_runs = stress_runs;
            res.avg_time_ms = avg_time * 1000;
            res.min_time_ms = min_time * 1000;
            res.max_time_ms = max_time * 1000;
            res.std_deviation = stddev * 1000;
            res.gflops = avg_gflops;
            res.memory_allocated_gb = memory_gb;
            res.passed_validation = passed;
            res.max_numerical_error = max_error;

            results.push_back(res);

            std::cout << "  Statistics:\n";
            std::cout << "    Time (ms):     Avg=" << std::fixed << std::setprecision(3) 
                      << avg_time * 1000 << " | Min=" << min_time * 1000 
                      << " | Max=" << max_time * 1000 << " | StdDev=" << stddev * 1000 << "\n";
            std::cout << "    GFLOPS:        Avg=" << avg_gflops << " | Min=" 
                      << min_gflops << " | Max=" << max_gflops << "\n";
            std::cout << "    Memory:        " << memory_gb << " GB allocated\n";
            std::cout << "    Validation:    " << (passed ? "PASSED" : "FAILED") 
                      << " (Max Error: " << max_error << ")\n";
            std::cout << "    Stability:     " << (stddev < avg_time * 0.1 ? "GOOD" : "VARIABLE") 
                      << " (CV: " << (stddev / avg_time) << ")\n\n";

            // Cleanup
            cudaFree(d_A.ptr);
            cudaFree(d_B.ptr);
            cudaFree(d_C_cublas.ptr);
            cudaFree(d_C_naive.ptr);
            delete[] h_A.ptr;
            delete[] h_B.ptr;
            delete[] h_C_cublas.ptr;
            delete[] h_C_naive.ptr;

        } catch (const std::exception& e) {
            std::cerr << "  ERROR: " << e.what() << " (likely out of memory)\n\n";
        }
    }

    // Write comprehensive results to CSV
    std::ofstream outfile("benchmark_stress_test_results.csv");
    outfile << "MatrixSize,NumRuns,AvgTime_ms,MinTime_ms,MaxTime_ms,StdDev_ms,AvgGFLOPS,MemoryAllocated_GB,PassedValidation,MaxNumericalError\n";
    for (const auto& res : results) {
        outfile << res.matrix_size << ","
                << res.num_runs << ","
                << res.avg_time_ms << ","
                << res.min_time_ms << ","
                << res.max_time_ms << ","
                << res.std_deviation << ","
                << res.gflops << ","
                << res.memory_allocated_gb << ","
                << (res.passed_validation ? "YES" : "NO") << ","
                << res.max_numerical_error << "\n";
    }
    outfile.close();

    std::cout << "\n✓ Stress test results written to: benchmark_stress_test_results.csv\n";

    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);

    return 0;
}
