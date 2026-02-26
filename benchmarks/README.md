# Benchmarks for pyBOED v2.0

Comprehensive benchmark suite for evaluating performance of dimensionality reduction methods in BOED.

## 📊 Available Benchmarks

### 1. `benchmark_dimension_reduction.py`

**Purpose**: Compare BOED performance with different reduction methods

**Tests**:

- Problem sizes: N = 100, 200, 500, 1000, 2000
- Methods: Full space, KLE, PCA
- Reduction ratios: 5x, 10x, 20x, 50x

**Metrics**:

- Design optimization time
- Memory usage
- Estimated speedup
- Energy captured

**Runtime**: ~5-10 minutes

**Usage**:

```bash
python benchmarks/benchmark_dimension_reduction.py
```

**Outputs**:

- `results/benchmarks/benchmark_dimension_reduction.json` - Raw data
- `results/benchmarks/benchmark_time_speedup.png` - Time and speedup plots
- `results/benchmarks/benchmark_memory.png` - Memory usage
- `results/benchmarks/benchmark_energy.png` - Energy preservation

---

### 2. `benchmark_forward_acceleration.py`

**Purpose**: Measure POD speedup for forward model evaluations

**Tests**:

- Spatial resolutions: N = 50, 100, 200, 500, 1000, 2000
- Time steps: 10, 50, 100, 200
- POD energy thresholds: 0.99, 0.999, 0.9999

**Metrics**:

- Offline construction time
- Online evaluation time
- Actual speedup factor
- Accuracy (L2 error, max error)
- Number of POD modes

**Runtime**: ~10-15 minutes

**Usage**:

```bash
python benchmarks/benchmark_forward_acceleration.py
```

**Outputs**:

- `results/benchmarks/benchmark_forward_acceleration.json` - Raw data
- `results/benchmarks/pod_construction.png` - Construction time & modes
- `results/benchmarks/pod_speedup_accuracy.png` - Speedup vs accuracy
- `results/benchmarks/pod_time_comparison.png` - Time comparison

---

### 3. `benchmark_posterior_inference.py`

**Purpose**: Compare posterior computation methods

**Tests**:

- Parameter dimensions: 50, 100, 200, 500, 1000
- Observation dimensions: 10, 20, 50
- LIS ranks: 5, 10, 20, 50

**Metrics**:

- Posterior computation time
- LIS construction time (amortized)
- Mean and covariance errors
- Information gain preservation
- Variance captured

**Runtime**: ~15-20 minutes

**Usage**:

```bash
python benchmarks/benchmark_posterior_inference.py
```

**Outputs**:

- `results/benchmarks/benchmark_posterior_inference.json` - Raw data
- `results/benchmarks/lis_time_speedup.png` - Time and speedup
- `results/benchmarks/lis_accuracy_variance.png` - Accuracy analysis
- `results/benchmarks/lis_memory.png` - Memory efficiency

---

## 🚀 Quick Start

### Run All Benchmarks

```bash
# Run all benchmarks sequentially
python benchmarks/benchmark_dimension_reduction.py
python benchmarks/benchmark_forward_acceleration.py
python benchmarks/benchmark_posterior_inference.py
```

**Total Runtime**: ~30-45 minutes

### Run Individual Benchmarks

```bash
# Just dimension reduction
python benchmarks/benchmark_dimension_reduction.py

# Just forward acceleration
python benchmarks/benchmark_forward_acceleration.py

# Just posterior inference
python benchmarks/benchmark_posterior_inference.py
```

---

## 📈 Expected Results

### Dimension Reduction Benchmark

- **Speedup**: 10-100x for reduction ratios 10-50x
- **Memory savings**: 80-95% for large problems
- **Energy preservation**: >99% with appropriate n_components

### Forward Acceleration Benchmark

- **Speedup**: 5-50x depending on problem size
- **Accuracy**: <1e-3 relative error typical
- **Modes needed**: 20-50 for 0.9999 energy threshold

### Posterior Inference Benchmark

- **Speedup**: 10-100x for LIS ranks 5-20
- **Accuracy**: <1e-3 relative error in posterior mean
- **Information preservation**: >95% with LIS rank 10-20

---

## 📊 Interpreting Results

### JSON Output Format

Each benchmark saves JSON with structure:

```json
{
  "method": "kle|pca|full|lis",
  "N": 1000,
  "reduction_ratio": 20,
  "design_time": 1.234,
  "memory_mb": 56.7,
  "estimated_speedup": 45.6,
  "energy_ratio": 0.9987,
  "success": true
}
```

### Key Metrics

**design_time / time_mean**:

- Time for design/posterior computation
- Lower is better

**speedup / estimated_speedup**:

- Ratio of full time / reduced time
- Higher is better
- Expected: 10-100x for typical reductions

**energy_ratio / variance_ratio**:

- Fraction of information preserved
- Range: 0-1
- Target: >0.99 for high fidelity

**rel_error / error_mean**:

- Relative error in solution
- Lower is better
- Typical: <1e-3 for good methods

---

## 🔧 Customization

### Modify Problem Sizes

Edit the configuration in each benchmark:

```python
# benchmark_dimension_reduction.py
problem_sizes = [100, 200, 500, 1000]  # Add/remove sizes
reduction_ratios = [5, 10, 20, 50]     # Modify ratios

# benchmark_forward_acceleration.py
sizes = [50, 100, 200, 500, 1000]      # Spatial resolutions
n_steps_list = [10, 50, 100]           # Time steps
thresholds = [0.99, 0.999, 0.9999]     # POD thresholds

# benchmark_posterior_inference.py
param_dims = [50, 100, 200, 500]       # Parameter dimensions
obs_dims = [10, 20, 50]                # Observation dimensions
lis_ranks = [5, 10, 20]                # LIS ranks
```

### Save Custom Results

```python
# Add custom output directory
output_dir = Path('results/benchmarks/custom')
save_and_visualize(results, output_dir)
```

### Add Custom Metrics

```python
# In benchmark functions, add to results dict
results = {
    # ... existing metrics ...
    'custom_metric': my_custom_value,
    'another_metric': another_value
}
```

---

## 🐛 Troubleshooting

### Out of Memory

- Reduce problem sizes
- Use smaller reduction ratios
- Run benchmarks individually

### Slow Execution

- Comment out large problem sizes
- Reduce number of trials (n_trials parameter)
- Skip full space benchmarks for N > 500

### Missing Dependencies

```bash
pip install psutil  # For memory measurements
```

---

## 📝 Benchmark Design Principles

### Problem Selection

- Cover realistic parameter ranges
- Include both easy and hard cases
- Test scaling behavior

### Metric Selection

- Time: wall-clock time (most relevant)
- Memory: peak RSS (realistic usage)
- Accuracy: multiple measures (mean, max, relative)
- Quality: information-theoretic (EIG, variance)

### Statistical Rigor

- Multiple trials (3-10) for timing
- Standard deviation reported
- Outlier handling
- Reproducible seeds

---

## 📖 References

### Methods

- **KLE**: Loève, M. (1977). Probability Theory
- **POD**: Lumley, J.L. (1967). Atmospheric Turbulence
- **LIS**: Cui et al. (2014). Likelihood-Informed Dimension Reduction

### BOED Theory

- Chaloner & Verdinelli (1995). Bayesian Experimental Design
- Ryan et al. (2016). A Review of Modern Computational Algorithms

---

## 🤝 Contributing

To add new benchmarks:

1. Follow naming convention: `benchmark_<topic>.py`
2. Include docstring with purpose and metrics
3. Save results to JSON
4. Generate visualization plots
5. Add summary statistics
6. Update this README

Example template:

```python
"""
Benchmark: [Purpose]
=====================

[Description]

Metrics:
- [Metric 1]
- [Metric 2]
"""

def run_benchmark_suite():
    # Configuration
    # Run tests
    # Return results
    pass

def save_and_visualize(results, output_dir):
    # Save JSON
    # Create plots
    pass

def main():
    results = run_benchmark_suite()
    save_and_visualize(results, Path('results/benchmarks'))
    print_summary(results)

if __name__ == "__main__":
    main()
```

---

**Last Updated**: February 2024
**Version**: 2.0.0
**Maintainer**: pyBOED Team
