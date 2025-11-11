cmake_minimum_required(VERSION 3.18)
project(TensorBench LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Path to Eigen
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen-3.4.0")

# Pick the right compute capability for *your* GPU:
# Examples:
#   61  -> Pascal (GTX 1080, etc.)
#   75  -> Turing (RTX 20xx)
#   86  -> Ampere (RTX 30xx)
#   89  -> Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 89)  # <-- change 75 to what your GPU actually is


# CUDA toolkit (for cuBLAS)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Library with MatrixFP32, MatrixFP16, utils (replaces the .o files)
# ------------------------------------------------------------------------------

add_library(matrix_objs STATIC
    src/MatrixFP32.cu
    src/MatrixFP16.cu
    src/utils.cu
)

# Equivalent to -dc (separable compilation)
set_target_properties(matrix_objs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(matrix_objs PRIVATE
    ${EIGEN_INCLUDE_DIR}
)

# Optional: ptxas verbose
target_compile_options(matrix_objs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-O3>
)

# ------------------------------------------------------------------------------
# 00_benchmark_cuBLAS.out
# ------------------------------------------------------------------------------

add_executable(bench_cublas
    test/00_benchmark_cuBLAS.cu
)

set_target_properties(bench_cublas PROPERTIES
    OUTPUT_NAME "00_benchmark_cuBLAS.out"
)

target_link_libraries(bench_cublas PRIVATE
    matrix_objs
    CUDA::cublas
)

target_compile_options(bench_cublas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# ------------------------------------------------------------------------------
# 01_benchmark_naive.out
# ------------------------------------------------------------------------------

add_executable(bench_naive
    src/naive_tensor_tgemm.cu
    test/01_benchmark_naive.cu
)

set_target_properties(bench_naive PROPERTIES
    OUTPUT_NAME "01_benchmark_naive.out"
)

# -arch=sm_86 only for this benchmark
target_compile_options(bench_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86;--ptxas-options=-v>
)

target_link_libraries(bench_naive PRIVATE
    matrix_objs
    CUDA::cublas
)

# ------------------------------------------------------------------------------
# 02_benchmark_mixed_precision.out (Complex Mixed-Precision Benchmark)
# ------------------------------------------------------------------------------

add_executable(bench_mixed_precision
    src/naive_tensor_tgemm.cu
    test/02_benchmark_mixed_precision.cu
)

set_target_properties(bench_mixed_precision PROPERTIES
    OUTPUT_NAME "02_benchmark_mixed_precision.out"
)

target_compile_options(bench_mixed_precision PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-O3>
)

target_link_libraries(bench_mixed_precision PRIVATE
    matrix_objs
    CUDA::cublas
)

# ------------------------------------------------------------------------------
# 03_benchmark_scaling.out (Scaling & Strong Scaling Analysis)
# ------------------------------------------------------------------------------

add_executable(bench_scaling
    src/naive_tensor_tgemm.cu
    test/03_benchmark_scaling.cu
)

set_target_properties(bench_scaling PROPERTIES
    OUTPUT_NAME "03_benchmark_scaling.out"
)

target_compile_options(bench_scaling PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-O3>
)

target_link_libraries(bench_scaling PRIVATE
    matrix_objs
    CUDA::cublas
)

# ------------------------------------------------------------------------------
# 04_benchmark_stress_test.out (Stress Test & Memory Hierarchy)
# ------------------------------------------------------------------------------

add_executable(bench_stress_test
    src/naive_tensor_tgemm.cu
    test/04_benchmark_stress_test.cu
)

set_target_properties(bench_stress_test PROPERTIES
    OUTPUT_NAME "04_benchmark_stress_test.out"
)

target_compile_options(bench_stress_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-O3>
)

target_link_libraries(bench_stress_test PRIVATE
    matrix_objs
    CUDA::cublas
)

# ------------------------------------------------------------------------------
# 05_benchmark_advanced_tensor_ops.out (Advanced Multi-Algorithm Analysis)
# 
# Most Complex Benchmark: 500+ lines
# - Multi-algorithm comparison (naive, cuBLAS, fused operations)
# - Roofline model performance analysis
# - Cache behavior estimation
# - Advanced statistical analysis with confidence intervals
# - Thermal throttling risk assessment
# - Power efficiency metrics (GFLOPS/Watt estimates)
# - 2 Benchmark Phases:
#   * Phase 1: Single Matrix Multiplication Comparison
#   * Phase 2: Fused Operations Analysis (A1*B1 + A2*B2)
# - CSV export for advanced analysis
# 
# Features:
# - 15 runs per matrix size for statistical significance
# - 95% confidence interval calculations
# - Roofline model ceiling computation
# - Thermal throttling risk estimation
# - Cache miss rate approximation
# - Memory bandwidth analysis
# - Peak efficiency percentage calculation
# 
# Output Files:
# - benchmark_advanced_metrics.csv (detailed per-operation metrics)
# - benchmark_roofline_model.csv (roofline analysis data)
# 
# Expected Runtime: 2-5 minutes depending on GPU and matrix sizes
# Memory Usage: ~4GB for 4096x4096 matrices
# 
# Performance Metrics Tracked:
# 1. Execution Time (ms) with variance/std deviation
# 2. GFLOPS and efficiency percentage
# 3. Memory bandwidth (GB/s)
# 4. Compute intensity (FLOPS/Byte)
# 5. Cache miss estimation
# 6. Thermal throttle risk
# 7. Roofline model ceiling
# 8. Performance bottleneck classification
# 
# Algorithm Implementations:
# 1. cuBLAS Optimized (tensor operations)
# 2. Naive Kernel (reference implementation)
# 3. Fused Operations (A*B + C*D)
# 
# GPU Specifications Tracked:
# - Peak compute performance (GFLOPS)
# - Peak memory bandwidth (GB/s)
# - Warp size and thread limits
# - Number of SMs (streaming multiprocessors)
# - Power consumption (TDP)
# 
# Analysis Outputs:
# - Algorithm performance ranking
# - Roofline model visualization data
# - Algorithm comparison summary
# - Statistical analysis with confidence intervals
# 
# Test Sizes: 256, 512, 1024, 2048, 4096 (configurable)
# Runs per Size: 15 (configurable for accuracy)
# Warmup Runs: 3 per configuration
# 
# This is the most comprehensive benchmark test in TensorBench
# 
# Suitable for:
# - Deep performance analysis
# - Comparative studies
# - Architecture evaluation
# - Optimization research
# - Performance tuning
# - Capability assessment
# 
# Author: TensorBench Advanced Framework
# Purpose: Comprehensive tensor operation performance characterization
# 
# ======================================================================

add_executable(bench_advanced_tensor_ops
    src/naive_tensor_tgemm.cu
    test/05_benchmark_advanced_tensor_ops.cu
)

set_target_properties(bench_advanced_tensor_ops PROPERTIES
    OUTPUT_NAME "05_benchmark_advanced_tensor_ops.out"
)

target_compile_options(bench_advanced_tensor_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-O3>
)

target_link_libraries(bench_advanced_tensor_ops PRIVATE
    matrix_objs
    CUDA::cublas
)

