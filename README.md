# Eigenvalue Computation Performance and Accuracy

## Overview
This project implements and compares three methods for computing eigenvalues of 3x3 symmetric positive semi-definite matrices:
1. **NumPy Implementation**: Using `np.linalg.eigvalsh` from the NumPy library
2. **Custom Implementation**: Pure Python implementation based on Trigonometric analytical solution for real roots
3. **CUDA Implementation**: GPU-accelerated version using CUDA

## Performance Analysis

### Test Configuration
- **Matrix Count**: 1,000,000 randomly generated matrices
- **Test Runs**: 5 independent runs
- **Matrix Type**: 3x3 symmetric positive semi-definite

### Timing Results

| Method   | Avg Time per Run (s) | Total Time (s) | Avg Time per Matrix (μs) |
|----------|---------------------|----------------|------------------------|
| numpy    | 0.400891           | 2.00445        | 4.009                 |
| custom   | 2.04138            | 10.2069        | 20.414                |
| cuda     | 0.032132           | 0.160661       | 0.321                 |

### Key Performance Observations
- **CUDA Implementation**: Achieved the best performance, significantly outperforming other methods
- **NumPy Implementation**: Shows good performance due to optimized BLAS/LAPACK libraries
- **Custom Implementation**: Slower due to pure Python implementation without low-level optimizations

## Accuracy Analysis

### Comparison Results

| Method 1 | Method 2 | Max Diff  | Avg Diff  | Success Rate |
|----------|----------|-----------|-----------|--------------|
| NumPy    | Custom   | 6.36e-13  | 1.1e-15   | 100.00%     |
| NumPy    | CUDA     | 6.37e-13  | 9.29e-16  | 100.00%     |
| Custom   | CUDA     | 7.73e-13  | 8.07e-16  | 100.00%     |

### Special Test Cases
The following edge cases were successfully tested:
- Diagonal matrix with distinct eigenvalues
- Identity matrix
- Zero matrix
- Matrix with repeated eigenvalues
- Nearly singular matrix

## Implementation Details

### Software Dependencies
- Python 3.9+
- NumPy 1.21.0+
- CUDA Toolkit 11.2+
- Pybind11 2.6.0+

### Installation
```bash
# Clone the repository
git clone https://github.com/username/eigenvalue-computation
cd eigenvalue-computation

# Build CUDA extension
python setup.py build_ext --inplace
```

### Usage Example
```python
import numpy as np
import eigenvalues_cuda

# Prepare input matrices (N x 3 x 3 symmetric matrices)
matrices = np.array([...], dtype=np.float64)

# Reshape for CUDA processing
matrices_flat = matrices.reshape(-1, 9)
eigenvalues = np.zeros((len(matrices), 3), dtype=np.float64)

# Compute eigenvalues
eigenvalues_cuda.compute_eigenvalues(matrices_flat, eigenvalues)
```

## Test Environment

## Visualization
A comparison plot of computation times.
![comparison](/timing_comparison.png)

## Future Development
1. **Extended Matrix Support**
   - Implementation for larger matrices (4x4, 5x5)
   - Support for non-symmetric matrices

2. **Performance Optimization**
   - Memory transfer optimization
   - Kernel optimization for different GPU architectures
   - Batch size optimization

3. **Feature Additions**
   - Eigenvector computation
   - Support for complex matrices
   - Additional numerical precision options

## Testing
To reproduce the performance results:
```bash
python test.py
```

## References
1. NumPy Linear Algebra Documentation: [numpy.linalg.eigvalsh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html)
2. CUDA Programming Guide: [NVIDIA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
3. Trigonometric solution for three real roots : [Link](https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions)

## Contact
[ovengurl@asu.edu]