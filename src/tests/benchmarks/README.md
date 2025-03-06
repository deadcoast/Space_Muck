# Space Muck Benchmarks

This directory contains benchmark scripts for measuring performance of various Space Muck components.

## Running Benchmarks

### Base Generator Benchmarks
```bash
# Run base generator benchmarks
python -m src.tests.benchmarks.benchmark_base_generator
```

### GPU Acceleration Benchmarks
```bash
# Run GPU acceleration benchmarks
python -m src.tests.benchmarks.benchmark_gpu_acceleration

# Run GPU noise generation benchmarks
python -m src.tests.benchmarks.benchmark_gpu_noise_generation

# Run comprehensive GPU benchmarks
python -m src.tests.benchmarks.benchmark_comprehensive_gpu

# Run comprehensive GPU benchmarks with main entry point
python -m src.tests.benchmarks.benchmark_comprehensive_gpu_main
```

### Parallel Processing Benchmarks
```bash
# Run parallel processing benchmarks
python -m src.tests.benchmarks.benchmark_parallel_processing
```

### Procedural Generation Benchmarks
```bash
# Run procedural generation benchmarks
python -m src.tests.benchmarks.benchmark_procedural_generation
```

## Benchmark Results

Benchmark results are typically displayed in the console and may include:
- Execution time comparisons
- Memory usage statistics
- Speedup factors for different implementations
- Performance comparisons across different input sizes

For visualization of benchmark results, many scripts will generate plots if matplotlib is available.
