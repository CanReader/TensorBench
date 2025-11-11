# 🚀 TensorBench: Advanced CUDA Tensor Operation Benchmarking Suite

<div align="center">

![CUDA](https://img.shields.io/badge/CUDA-12.0+-green?style=flat-square&logo=nvidia)
![C++](https://img.shields.io/badge/C++-17-blue?style=flat-square&logo=cplusplus)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

**Comprehensive benchmarking framework for GPU tensor operations with roofline model analysis, multi-algorithm comparison, and advanced performance characterization.**

[Features](#-features) • [Quick Start](#-quick-start) • [Benchmarks](#-benchmark-suite) • [Results](#-output--analysis) • [Architecture](#-architecture)

</div>

---

## 📋 Overview

**TensorBench** is a production-grade CUDA benchmarking suite designed for comprehensive performance analysis of tensor operations on NVIDIA GPUs. It provides deep insights into GPU utilization, memory hierarchy behavior, and comparative performance across multiple algorithms.

### Use Cases
- 🔍 **Performance Profiling**: Detailed analysis of tensor operation performance
- 🏗️ **Architecture Evaluation**: Compare different implementation strategies
- 📊 **Optimization Research**: Identify bottlenecks and optimization opportunities
- 🎯 **GPU Capability Assessment**: Understand your hardware's strengths and limitations
- 📈 **Comparative Studies**: Benchmark naive vs. optimized implementations

---

## ✨ Features

### 🎯 Multi-Algorithm Comparison
- **cuBLAS Optimized**: Highly-tuned vendor library implementation
- **Naive Kernel**: Reference implementation for correctness validation
- **Fused Operations**: Advanced multi-operation kernels

### 📊 Advanced Performance Analysis
- **Roofline Model**: Theoretical performance ceiling computation
- **Arithmetic Intensity**: FLOPS/Byte analysis for memory vs. compute bottleneck classification
- **Cache Behavior**: L1/L2 cache miss estimation
- **Memory Bandwidth**: Real-time measurement and analysis
- **Statistical Analysis**: Mean, variance, standard deviation, 95% confidence intervals

### 🔬 Deep Profiling Capabilities
- **Thermal Throttling**: Risk assessment based on sustained execution
- **Efficiency Metrics**: Peak efficiency percentage calculation
- **Variance Analysis**: Execution time stability tracking
- **Power Efficiency**: GFLOPS/Watt estimation
- **GPU Properties**: Full device capability reporting

### 📁 Comprehensive Output
- CSV exports for advanced analysis
- Multi-phase benchmarking (single ops, fused ops, batch processing)
- Detailed performance logs with human-readable formatting

---

## 🏗️ Architecture

### Project Structure
```
TensorBench/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/
│   ├── MatrixFP16.cuh          # FP16 matrix class definition
│   ├── MatrixFP32.cuh          # FP32 matrix class definition
│   ├── naive_tensor_tgemm.cuh  # Naive GEMM kernel header
│   └── utils.cuh               # Utility functions
├── src/
│   ├── MatrixFP16.cu           # FP16 matrix implementation
│   ├── MatrixFP32.cu           # FP32 matrix implementation
│   ├── naive_tensor_tgemm.cu   # Naive GEMM kernel implementation
│   └── utils.cu                # Utility implementations
├── test/
│   ├── 00_benchmark_cuBLAS.cu                 # Test 1: cuBLAS Baseline
│   ├── 01_benchmark_naive.cu                  # Test 2: Naive Implementation
│   ├── 02_benchmark_mixed_precision.cu        # Test 3: Mixed Precision Analysis
│   ├── 03_benchmark_scaling.cu                # Test 4: Strong Scaling
│   ├── 04_benchmark_stress_test.cu            # Test 5: Stress Testing
│   └── 05_benchmark_advanced_tensor_ops.cu    # Test 6: Advanced Analysis (500+ lines)
└── build/                      # CMake build output
    └── *.out                   # Compiled executables
```

---

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with compute capability 6.1+ (Pascal or newer)
- **CUDA Toolkit** 12.0 or later
- **CMake** 3.18+
- **GCC/G++** 9+ or **LLVM/Clang** 10+

### Installation & Build

For the simplest setup, run the appropriate command below. These scripts automatically handle the CMake configuration and compilation.

> ### **🐧 Linux/macOS**
> Run:
> ```bash
> ./build.sh
> ```

> ### **🪟 Windows**
> Run:
> ```bash
> build.bat
> ```

**or**


1. **Clone and navigate to the project:**
```bash
cd TensorBench
```

2. **Configure with CMake:**
```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

3. **Build all benchmarks:**
```bash
cmake --build build -j$(nproc)
```

4. **Or build specific tests:**
```bash
# Build only advanced tensor ops benchmark
cmake --build build --target bench_advanced_tensor_ops

# Build only scaling analysis
cmake --build build --target bench_scaling

# List all targets
cmake --build build --target help
```

### Running Benchmarks

After building, run individual benchmarks from the `build/` directory:

```bash
cd build

# Test 1: cuBLAS Baseline Performance
./00_benchmark_cuBLAS.out

# Test 2: Naive Kernel Comparison
./01_benchmark_naive.out

# Test 3: Mixed Precision Analysis
./02_benchmark_mixed_precision.out

# Test 4: Strong Scaling Analysis
./03_benchmark_scaling.out

# Test 5: Stress Testing
./04_benchmark_stress_test.out

# Test 6: Advanced Tensor Operations (Most Comprehensive)
./05_benchmark_advanced_tensor_ops.out
```

---

## 📊 Benchmark Suite

### Test 1: cuBLAS Baseline (`00_benchmark_cuBLAS.cu`)
**Purpose**: Establish performance baseline with vendor-optimized library

| Aspect | Details |
|--------|---------|
| **Sizes** | 128, 256, 512, 1024, 2048, 4096 |
| **Runs** | 10 per size |
| **Algorithm** | cuBLAS GemmEx with tensor operations |
| **Output** | GFLOPS, execution time |
| **Use Case** | Reference performance ceiling |

---

### Test 2: Naive Implementation (`01_benchmark_naive.cu`)
**Purpose**: Reference kernel for correctness validation

| Aspect | Details |
|--------|---------|
| **Sizes** | 128, 256, 512, 1024, 2048, 4096 |
| **Runs** | 10 per size |
| **Algorithm** | Custom naive GEMM kernel |
| **Validation** | Assert correctness against cuBLAS |
| **Output** | GFLOPS comparison, error analysis |
| **Use Case** | Correctness verification, optimization baseline |

---

### Test 3: Mixed Precision Analysis (`02_benchmark_mixed_precision.cu`) ⭐
**Purpose**: Comprehensive mixed-precision performance comparison

| Aspect | Details |
|--------|---------|
| **Sizes** | 128, 256, 512, 1024, 2048, 4096, **8192** |
| **Precision** | FP16 (input) × FP16 (input) → FP32 (output) |
| **Batches** | 3 independent benchmark runs |
| **Runs** | 5 per batch |
| **Statistics** | Mean time, GFLOPS, speedup metrics |
| **Output** | `benchmark_results.csv` with detailed metrics |
| **Features** | GPU device properties, warmup runs |
| **Use Case** | Mixed-precision optimization analysis |

**Key Metrics:**
- Time per operation (ms)
- GFLOPS achieved
- Speedup relative to naive implementation
- Numerical accuracy validation

---

### Test 4: Strong Scaling Analysis (`03_benchmark_scaling.cu`) 📈
**Purpose**: Analyze batch processing and strong scaling behavior

| Aspect | Details |
|--------|---------|
| **Sizes** | 256, 512, 1024, 2048 |
| **Batch Sizes** | 1, 2, 4, 8, 16 matrices |
| **Modes** | Sequential vs. queue-based execution |
| **Metrics** | Throughput (matrices/sec), memory bandwidth (GB/s) |
| **Output** | `benchmark_scaling_results.csv` |
| **Analysis** | Speedup, efficiency, bottleneck identification |

**Key Insights:**
- Batch processing efficiency
- Strong scaling characteristics
- Memory bandwidth utilization
- Queue vs. sequential overhead

---

### Test 5: Stress Testing (`04_benchmark_stress_test.cu`) 🔥
**Purpose**: Push GPU to limits; analyze maximum problem sizes and stability

| Aspect | Details |
|--------|---------|
| **Max Size** | **12288 × 12288** matrices |
| **Runs** | 20-50 per size (adaptive) |
| **Focus** | Execution time variance, stability |
| **Metrics** | Min, max, avg GFLOPS; variance analysis |
| **Output** | CSV with detailed statistics |
| **Safety** | Handles out-of-memory gracefully |
| **Use Case** | Maximum capacity planning, thermal limits |

**Variance Analysis:**
- Identifies thermal throttling
- Detects performance degradation
- Measures consistency across runs

---

### Test 6: Advanced Tensor Operations (`05_benchmark_advanced_tensor_ops.cu`) 🌟
**Purpose**: Most comprehensive analysis with roofline model and multi-algorithm comparison
**Lines of Code**: 650+ lines

| Aspect | Details |
|--------|---------|
| **Sizes** | 256, 512, 1024, 2048, 4096 |
| **Runs** | 15 per size (high statistical significance) |
| **Phases** | 2 benchmark phases |
| **Algorithms** | 3 implementations (cuBLAS, Naive, Fused) |
| **Output** | Multiple CSV files for advanced analysis |

#### Phase 1: Single Matrix Multiplication Comparison
- Compares cuBLAS vs. naive kernel
- Statistical analysis with 95% confidence intervals
- Roofline model analysis
- Cache miss estimation
- Thermal throttling risk assessment

#### Phase 2: Fused Operations Analysis
- Tests combined operations: C = A₁B₁ + A₂B₂
- Kernel fusion efficiency
- Memory access pattern optimization

#### Metrics Tracked:
```
Per Operation:
├── Execution Time (ms) + Variance
├── GFLOPS + Efficiency %
├── Memory Bandwidth (GB/s)
├── Compute Intensity (FLOPS/Byte)
├── Cache Miss Estimation
└── Thermal Throttle Risk (0.0-1.0)

Roofline Model:
├── Compute Intensity
├── Achieved GFLOPS
├── Peak Compute (Theoretical)
├── Peak Memory Bandwidth (Theoretical)
└── Bottleneck Classification (Compute vs. Memory)

Confidence Intervals:
├── 95% CI for mean execution time
├── Statistical significance
└── Accuracy bounds
```

#### GPU Specifications Reported:
- Peak compute (GFLOPS)
- Peak memory bandwidth (GB/s)
- Warp size
- Max threads per block
- Number of SMs
- TDP estimate

#### CSV Outputs:
- `benchmark_advanced_metrics.csv` - Per-operation detailed metrics
- `benchmark_roofline_model.csv` - Roofline analysis data

#### Expected Runtime:
- Small GPUs: 2-3 minutes
- Large GPUs (RTX 4090): 3-5 minutes

---

## 📈 Output & Analysis

### CSV Export Format

All benchmarks export detailed metrics to CSV for further analysis:

**benchmark_results.csv** (Mixed Precision):
```
MatrixSize,Batch,cuBLAS_Time_ms,Naive_Time_ms,cuBLAS_GFLOPS,Naive_GFLOPS,Speedup,MaxError,AvgError
128,1,0.123456,0.987654,123.45,15.67,8.03,1.2e-5,3.4e-6
256,1,0.234567,1.234567,234.56,18.90,5.25,1.5e-5,4.2e-6
...
```

**benchmark_scaling_results.csv** (Scaling Analysis):
```
MatrixSize,BatchSize,SequentialTime_ms,BatchTime_ms,SequentialGFLOPS,BatchGFLOPS,Speedup,Throughput_matrices_per_sec,MemoryBandwidth_GB_s
256,1,0.123,0.123,1234.5,1234.5,1.00,8130.08,987.65
256,2,0.246,0.180,617.3,841.5,1.37,10869.57,1289.45
...
```

**benchmark_advanced_metrics.csv** (Advanced):
```
MatrixSize,Algorithm,ExecutionTime_ms,GFLOPS,Efficiency_%,MemoryBandwidth_GB_s,ComputeIntensity,CacheMisses,ThrottleRisk
256,cuBLAS,0.123,1234.56,85.3,123.45,16.78,1024,0.15
256,Naive,0.456,333.33,23.0,45.67,16.78,4096,0.22
...
```

### Python Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('benchmark_advanced_metrics.csv')

# Plot GFLOPS vs. Matrix Size
plt.figure(figsize=(12, 6))
for algo in df['Algorithm'].unique():
    subset = df[df['Algorithm'] == algo]
    plt.plot(subset['MatrixSize'], subset['GFLOPS'], marker='o', label=algo)
plt.xlabel('Matrix Size')
plt.ylabel('GFLOPS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.title('Tensor Operation Performance Scaling')
plt.savefig('performance_scaling.png', dpi=300)
plt.show()
```

---

## 🔧 Configuration

### Adjusting GPU Architecture

Edit `CMakeLists.txt` line 15 to match your GPU:

```cmake
# Compute Capability Reference:
# 61  -> Pascal (GTX 1080, GTX 1070, etc.)
# 75  -> Turing (RTX 2080, RTX 2070, etc.)
# 86  -> Ampere (RTX 3090, RTX 3080, A100, etc.)
# 89  -> Ada (RTX 4090, RTX 4080, RTX 4070 Ti, etc.)

set(CMAKE_CUDA_ARCHITECTURES 89)  # <-- CHANGE THIS
```

### Adjusting Benchmark Parameters

Each test allows configuration through source code (test files):

**Matrix Sizes** (in each test file):
```cuda
int mat_sizes[] = {256, 512, 1024, 2048, 4096};  // Modify as needed
```

**Number of Runs**:
```cuda
int runs_per_batch = 5;   // Increase for more statistical precision
```

**Batch Sizes** (scaling test):
```cuda
int batch_sizes[] = {1, 2, 4, 8, 16};  // Modify batch configurations
```

---

## 📊 Understanding Roofline Model

The roofline model provides a theoretical performance ceiling based on:

1. **Arithmetic Intensity (AI)**: FLOPS per byte of memory transferred
2. **Peak Compute**: Maximum GFLOPS the GPU can achieve
3. **Peak Memory BW**: Maximum memory bandwidth available

**Performance Ceiling = min(Peak Compute, AI × Peak Memory BW)**

### Classification:
- **Memory-Bound**: Performance limited by memory bandwidth
- **Compute-Bound**: Performance limited by compute capacity

TensorBench automatically classifies each operation and suggests optimization directions.

---

## 🎯 Performance Optimization Tips

Based on TensorBench results, consider:

### If Memory-Bound:
- ✅ Increase tile size
- ✅ Improve cache locality
- ✅ Use mixed precision (FP16 inputs)
- ✅ Fuse multiple operations

### If Compute-Bound:
- ✅ Increase parallelism
- ✅ Improve instruction-level parallelism
- ✅ Use tensor cores (Turing+)
- ✅ Optimize register usage

### General:
- ✅ Monitor thermal throttling warnings
- ✅ Analyze variance for stability
- ✅ Compare against roofline ceiling
- ✅ Validate numerical accuracy

---

## 🖥️ System Requirements

### Minimum
- NVIDIA GPU: Compute Capability 6.1+ (GTX 1080 or newer)
- CUDA Toolkit: 11.0+
- RAM: 8GB
- Storage: 1GB

### Recommended
- NVIDIA GPU: Compute Capability 7.0+ (Turing+)
- CUDA Toolkit: 12.0+
- RAM: 16GB
- Storage: 2GB (for large benchmark runs)

### Tested Platforms
- ✅ Arch Linux-6.17.7 (x86_64)
- ✅ CUDA 13.0
- ✅ RTX 4050

---

## 🐛 Troubleshooting

### Build Errors

**"cuda_fp16.h not found"**
```bash
# Update your CUDA include path in .vscode/c_cpp_properties.json
# Or regenerate CMake configuration:
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

**"Cannot find cuBLAS"**
```bash
# Ensure CUDA toolkit is properly installed:
nvcc --version

# If not found, install or set CUDA path:
export CUDA_PATH=/usr/local/cuda
cmake -S . -B build
```

**Compilation fails on specific architecture**
```bash
# Check your GPU's compute capability:
nvidia-smi -q | grep -i "Compute Capability"

# Update CMakeLists.txt with correct value
set(CMAKE_CUDA_ARCHITECTURES 89)
```

### Runtime Issues

**Out of Memory**
- Reduce matrix size in test files
- Close other GPU applications
- Use `nvidia-smi` to check VRAM usage

**Thermal Throttling**
- Allow GPU cooling period between runs
- Reduce problem sizes
- Monitor with `nvidia-smi dmon`

**Inconsistent Results**
- Run benchmarks multiple times
- Check system background processes
- Verify power management settings
- Review confidence intervals in output

---

## 📚 Output Interpretation

### GFLOPS
- **Good**: > 80% of peak GPU GFLOPS
- **Acceptable**: 50-80% of peak
- **Poor**: < 50% indicates optimization opportunity

### Memory Bandwidth
- Check against theoretical peak
- High utilization (>90%) suggests memory optimization needed

### Variance
- **Low Variance**: Stable, consistent performance
- **High Variance**: May indicate thermal throttling or system interference

### Efficiency %
- **90-100%**: Excellent
- **70-90%**: Good
- **50-70%**: Fair, consider optimization
- **<50%**: Poor, significant optimization opportunity

---

## 📖 References & Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Roofline Model Paper](https://people.eecs.berkeley.edu/~sameh/SC06_paper.pdf)
- [NVIDIA GPU Architecture](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Additional optimization algorithms
- Extended GPU architecture support
- Performance analysis tools
- Documentation improvements
- Bug reports and fixes

---

## 📝 Citation

If you use TensorBench in your research, please cite:

```bibtex
@software{tensorbench2024,
  title={TensorBench: Advanced CUDA Tensor Operation Benchmarking Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TensorBench}
}
```

---

## ❓ FAQ

**Q: What GPU do I need?**
A: Any NVIDIA GPU with compute capability 6.1 or higher (GTX 1080 or newer). Newer GPUs (Turing+) provide better mixed-precision support.

**Q: How long do benchmarks take?**
A: 
- Quick tests (00-02): 30 seconds - 2 minutes
- Medium tests (03-04): 1-3 minutes
- Full suite (05): 3-5 minutes

**Q: Can I modify matrix sizes?**
A: Yes! Edit the test files to adjust `mat_sizes[]` array. Larger sizes require more VRAM.

**Q: How do I interpret results?**
A: Compare GFLOPS against your GPU's theoretical peak. Use roofline model to identify bottlenecks.

**Q: Why is my performance lower than expected?**
A: Check thermal throttling risk, compare against roofline ceiling, verify no system processes are interfering.

---

<div align="center">

**[⬆ Back to Top](#-tensorbench-advanced-cuda-tensor-operation-benchmarking-suite)**

Made with ❤️ for GPU performance enthusiasts and researchers

</div>
