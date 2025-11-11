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

// Structure to hold scaling analysis results
struct ScalingResult {
    int matrix_size;
    int num_iterations;
    double sequential_time_ms;
    double batch_time_ms;
    double sequential_gflops;
    double batch_gflops;
    double batch_speedup;
    double throughput_matrices_per_sec;
    double memory_bandwidth_gbs;
};

// Compute GFLOPS
double compute_gflops(int n, int num_ops, double time_seconds) {
    return (2.0 * n * n * n * num_ops * 1e-9) / time_seconds;
}

// Compute memory bandwidth (GB/s)
// For C = A * B: 3*n^2 reads + 1*n^2 writes (all in FP16 except C in FP32)
double compute_memory_bandwidth(int n, int num_ops, double time_seconds) {
    // Each matrix element is 2 bytes (FP16) or 4 bytes (FP32)
    // Read: 2*n^2 (A) + 2*n^2 (B) = 4*n^2 bytes
    // Write: 4*n^2 (C) = 4*n^2 bytes
    // Total: 8*n^2 per multiplication
    double bytes_transferred = 8.0 * n * n * num_ops; // in bytes
    double time_seconds_total = time_seconds;
    return (bytes_transferred / (1e9 * time_seconds_total)); // GB/s
}

// Batch processing kernel launcher with loop unrolling
void batch_sequential_multiply(cublasHandle_t handle, 
                               std::vector<MatrixFP16*>& d_A_batch,
                               std::vector<MatrixFP16*>& d_B_batch,
                               std::vector<MatrixFP32*>& d_C_batch,
                               int batch_size, int n) {
    for (int i = 0; i < batch_size; i++) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublas_check(cublasGemmEx(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, n, n,
                                &alpha,
                                d_B_batch[i]->ptr, CUDA_R_16F, n,
                                d_A_batch[i]->ptr, CUDA_R_16F, n,
                                &beta,
                                d_C_batch[i]->ptr, CUDA_R_32F, n,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaDeviceSynchronize();
}

// Batch processing with queue-like execution (all at once)
void batch_queue_multiply(cublasHandle_t handle, 
                          std::vector<MatrixFP16*>& d_A_batch,
                          std::vector<MatrixFP16*>& d_B_batch,
                          std::vector<MatrixFP32*>& d_C_batch,
                          int batch_size, int n) {
    // Issue all kernels without explicit synchronization
    for (int i = 0; i < batch_size; i++) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublas_check(cublasGemmEx(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, n, n,
                                &alpha,
                                d_B_batch[i]->ptr, CUDA_R_16F, n,
                                d_A_batch[i]->ptr, CUDA_R_16F, n,
                                &beta,
                                d_C_batch[i]->ptr, CUDA_R_32F, n,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaDeviceSynchronize();
}

int main(int argc, char const *argv[])
{
    std::cout << "\n╔════════════════════════════════════════════╗\n";
    std::cout << "║  TensorBench - Scaling Analysis Benchmark  ║\n";
    std::cout << "║  Strong Scaling & Batch Processing         ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";

    // Test parameters
    int mat_sizes[] = {256, 512, 1024, 2048};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);
    int batch_sizes[] = {1, 2, 4, 8, 16};
    int n_batch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));

    std::vector<ScalingResult> results;

    std::cout << "Starting scaling analysis...\n\n";

    for (int mat_idx = 0; mat_idx < n_sizes; mat_idx++) {
        int n = mat_sizes[mat_idx];
        std::cout << "Matrix Size: " << n << " x " << n << "\n";
        std::cout << std::string(50, '-') << "\n";

        for (int batch_idx = 0; batch_idx < n_batch_sizes; batch_idx++) {
            int batch_size = batch_sizes[batch_idx];

            // Allocate batch of matrices using pointers
            std::vector<MatrixFP16*> h_A_batch(batch_size);
            std::vector<MatrixFP16*> h_B_batch(batch_size);
            std::vector<MatrixFP32*> h_C_batch(batch_size);
            std::vector<MatrixFP16*> d_A_batch(batch_size);
            std::vector<MatrixFP16*> d_B_batch(batch_size);
            std::vector<MatrixFP32*> d_C_batch(batch_size);

            // Initialize all matrices
            for (int i = 0; i < batch_size; i++) {
                h_A_batch[i] = new MatrixFP16(n, n, false);
                h_B_batch[i] = new MatrixFP16(n, n, false);
                h_C_batch[i] = new MatrixFP32(n, n, false);
                random_init_mat(*h_A_batch[i], -5, 5);
                random_init_mat(*h_B_batch[i], -5, 5);
                init_mat(*h_C_batch[i], 0.0f);

                d_A_batch[i] = new MatrixFP16(n, n, true);
                d_B_batch[i] = new MatrixFP16(n, n, true);
                d_C_batch[i] = new MatrixFP32(n, n, true);

                h_A_batch[i]->copy_to_device(*d_A_batch[i]);
                h_B_batch[i]->copy_to_device(*d_B_batch[i]);
                h_C_batch[i]->copy_to_device(*d_C_batch[i]);
            }
            cudaDeviceSynchronize();

            // Warmup
            batch_sequential_multiply(handle, d_A_batch, d_B_batch, d_C_batch, batch_size, n);

            // Benchmark sequential execution
            int sequential_runs = 20;
            cudaEventRecord(beg);
            for (int run = 0; run < sequential_runs; run++) {
                batch_sequential_multiply(handle, d_A_batch, d_B_batch, d_C_batch, batch_size, n);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, beg, end);
            double sequential_time = elapsed_time / 1000.0 / sequential_runs; // Average per batch

            // Benchmark queue execution (all kernels at once)
            int queue_runs = 20;
            cudaEventRecord(beg);
            for (int run = 0; run < queue_runs; run++) {
                batch_queue_multiply(handle, d_A_batch, d_B_batch, d_C_batch, batch_size, n);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, beg, end);
            double batch_time = elapsed_time / 1000.0 / queue_runs; // Average per batch

            double sequential_gflops = compute_gflops(n, batch_size, sequential_time);
            double batch_gflops = compute_gflops(n, batch_size, batch_time);
            double speedup = sequential_time / batch_time;
            double throughput = batch_size / batch_time; // matrices per second
            double memory_bandwidth = compute_memory_bandwidth(n, batch_size, batch_time);

            ScalingResult res;
            res.matrix_size = n;
            res.num_iterations = batch_size;
            res.sequential_time_ms = sequential_time * 1000;
            res.batch_time_ms = batch_time * 1000;
            res.sequential_gflops = sequential_gflops;
            res.batch_gflops = batch_gflops;
            res.batch_speedup = speedup;
            res.throughput_matrices_per_sec = throughput;
            res.memory_bandwidth_gbs = memory_bandwidth;

            results.push_back(res);

            std::cout << "  Batch Size: " << std::setw(2) << batch_size 
                      << " | Sequential: " << std::fixed << std::setprecision(3) 
                      << sequential_gflops << " GFLOPS | Queue: " 
                      << batch_gflops << " GFLOPS | Speedup: " 
                      << speedup << "x | Throughput: " 
                      << throughput << " matrices/s | BW: " 
                      << memory_bandwidth << " GB/s\n";

            // Cleanup
            for (int i = 0; i < batch_size; i++) {
                delete d_A_batch[i];
                delete d_B_batch[i];
                delete d_C_batch[i];
                delete h_A_batch[i];
                delete h_B_batch[i];
                delete h_C_batch[i];
            }
        }
        std::cout << "\n";
    }

    // Write results to CSV
    std::ofstream outfile("benchmark_scaling_results.csv");
    outfile << "MatrixSize,BatchSize,SequentialTime_ms,BatchTime_ms,SequentialGFLOPS,BatchGFLOPS,Speedup,Throughput_matrices_per_sec,MemoryBandwidth_GB_s\n";
    for (const auto& res : results) {
        outfile << res.matrix_size << ","
                << res.num_iterations << ","
                << res.sequential_time_ms << ","
                << res.batch_time_ms << ","
                << res.sequential_gflops << ","
                << res.batch_gflops << ","
                << res.batch_speedup << ","
                << res.throughput_matrices_per_sec << ","
                << res.memory_bandwidth_gbs << "\n";
    }
    outfile.close();

    std::cout << "\n✓ Results written to: benchmark_scaling_results.csv\n";

    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);

    return 0;
}
