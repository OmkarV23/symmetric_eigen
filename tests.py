import time
import numpy as np
import eigenvalues_cuda
from typing import Tuple, List
from dataclasses import dataclass
from tabulate import tabulate
import matplotlib.pyplot as plt

@dataclass
class TestCase:
    """Class for holding test matrices and their properties"""
    name: str
    matrix: np.ndarray
    description: str

@dataclass
class TimingResult:
    """Class for holding timing results"""
    method: str
    total_time: float         # Total time across all runs
    avg_time_per_run: float   # Average time per run
    avg_time_per_matrix: float  # Average time per matrix per run
    matrices_count: int

@dataclass
class AccuracyResult:
    """Class for holding accuracy comparison results"""
    method1: str
    method2: str
    max_diff: float
    avg_diff: float
    success_rate: float
    failed_indices: List[int]

def generate_symmetric_psd_matrix() -> np.ndarray:
    """Generate a random 3x3 symmetric positive semi-definite matrix"""
    A = np.random.rand(3, 3)
    return np.dot(A, A.T)

def generate_special_cases() -> List[TestCase]:
    """Generate special test cases for eigenvalue computation"""
    test_cases = []
    
    # Diagonal matrix
    diagonal = np.diag([1.0, 2.0, 3.0])
    test_cases.append(TestCase("diagonal", diagonal, "Diagonal matrix with distinct eigenvalues"))
    
    # Identity matrix
    identity = np.eye(3)
    test_cases.append(TestCase("identity", identity, "Identity matrix"))
    
    # Zero matrix
    zero = np.zeros((3, 3))
    test_cases.append(TestCase("zero", zero, "Zero matrix"))
    
    # Matrix with repeated eigenvalues
    repeated = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    test_cases.append(TestCase("repeated", repeated, "Matrix with repeated eigenvalues"))
    
    # Nearly singular matrix
    nearly_singular = np.array([[1e-10, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_cases.append(TestCase("nearly_singular", nearly_singular, "Nearly singular matrix"))
    
    return test_cases

def compute_eigenvalues_custom(A: np.ndarray) -> np.ndarray:
    """Compute eigenvalues using custom implementation"""
    p1 = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
    if p1 == 0:
        eig1, eig2, eig3 = A[0, 0], A[1, 1], A[2, 2]
    else:
        q = np.trace(A) / 3
        p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2 * p1
        p = np.sqrt(p2 / 6)
        B = (1 / p) * (A - q * np.eye(3))
        r = np.linalg.det(B) / 2

        r = max(min(r, 1), -1)  # Clamp r to [-1, 1] to avoid numerical issues

        phi = np.arccos(r) / 3

        eig1 = q + 2 * p * np.cos(phi)
        eig3 = q + 2 * p * np.cos(phi + (2 * np.pi / 3))
        eig2 = 3 * q - eig1 - eig3

    return np.sort(np.array([eig1, eig2, eig3]))

def run_timing_test(matrices: np.ndarray, num_runs: int = 5) -> Tuple[TimingResult, TimingResult, TimingResult]:
    """Run timing tests for all three methods"""
    results = []
    num_matrices = len(matrices)

    for method in ['numpy', 'custom', 'cuda']:
        total_time = 0
        for _ in range(num_runs):
            if method == 'numpy':
                start = time.time()
                for matrix in matrices:
                    np.linalg.eigvalsh(matrix)
                end = time.time()
            elif method == 'custom':
                start = time.time()
                for matrix in matrices:
                    compute_eigenvalues_custom(matrix)
                end = time.time()
            else:  # cuda
                eigenvalues = np.zeros((num_matrices, 3), dtype=matrices.dtype)
                matrices_flat = matrices.reshape(-1, 9)  # Flatten the matrices
                start = time.time()
                eigenvalues_cuda.compute_eigenvalues(matrices_flat, eigenvalues)
                end = time.time()
            
            total_time += end - start

        avg_time_per_run = total_time / num_runs
        avg_time_per_matrix = avg_time_per_run / num_matrices
        results.append(TimingResult(method, total_time, avg_time_per_run, avg_time_per_matrix, num_matrices))
    
    return tuple(results)

def compare_accuracy(matrices: np.ndarray, atol: float = 1e-6) -> List[AccuracyResult]:
    """Compare accuracy between all three methods"""
    results = []

    # Compute eigenvalues using all methods
    numpy_eigs = np.array([np.sort(np.linalg.eigvalsh(m)) for m in matrices])
    custom_eigs = np.array([compute_eigenvalues_custom(m) for m in matrices])
    cuda_eigs = np.zeros((len(matrices), 3), dtype=matrices.dtype)
    matrices_flat = matrices.reshape(-1, 9)  # Flatten the matrices
    eigenvalues_cuda.compute_eigenvalues(matrices_flat, cuda_eigs)
    cuda_eigs = np.sort(cuda_eigs, axis=1)

    # Compare pairs
    methods = [
        ("NumPy", "Custom", numpy_eigs, custom_eigs),
        ("NumPy", "CUDA", numpy_eigs, cuda_eigs),
        ("Custom", "CUDA", custom_eigs, cuda_eigs)
    ]

    for method1, method2, eigs1, eigs2 in methods:
        diffs = np.abs(eigs1 - eigs2)
        max_diff = np.max(diffs)
        avg_diff = np.mean(diffs)
        failed_indices = np.where(np.any(diffs > atol, axis=1))[0].tolist()
        success_rate = (len(matrices) - len(failed_indices)) / len(matrices) * 100

        results.append(AccuracyResult(
            method1=method1,
            method2=method2,
            max_diff=max_diff,
            avg_diff=avg_diff,
            success_rate=success_rate,
            failed_indices=failed_indices
        ))

    return results

def plot_timing_comparison(timing_results: List[TimingResult]):
    """Plot timing comparison between methods"""
    methods = [result.method for result in timing_results]
    avg_times = [result.avg_time_per_run for result in timing_results]
    total_times = [result.total_time for result in timing_results]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, avg_times, width, label='Avg Time per Run')
    plt.bar(x + width/2, total_times, width, label='Total Time')

    plt.xticks(x, methods)
    plt.title('Computation Time by Method')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('timing_comparison.png')
    plt.close()

def main():
    # Test parameters
    num_matrices = 100000
    num_timing_runs = 5
    atol = 1e-6

    # Generate test matrices
    print("\n1. Generating test matrices...")
    random_matrices = np.array([generate_symmetric_psd_matrix() 
                              for _ in range(num_matrices)], dtype=np.float64)
    special_cases = generate_special_cases()

    # Run timing tests
    print("\n2. Running timing tests...")
    numpy_timing, custom_timing, cuda_timing = run_timing_test(
        random_matrices, num_timing_runs)

    # Run accuracy comparison
    print("\n3. Running accuracy tests...")
    accuracy_results = compare_accuracy(random_matrices, atol)

    # Test special cases
    print("\n4. Testing special cases...")
    special_case_results = []
    for case in special_cases:
        matrix_batch = np.array([case.matrix], dtype=np.float64)  # Ensure dtype is consistent
        case_accuracy = compare_accuracy(matrix_batch, atol)
        special_case_results.append((case, case_accuracy))

    # Generate reports
    print("\n=== Timing Results ===")
    timing_table = [[
        r.method,
        f"{r.avg_time_per_run:.6f}",
        f"{r.total_time:.6f}",
        f"{r.avg_time_per_matrix * 1e6:.3f}"  # Convert to microseconds
    ] for r in (numpy_timing, custom_timing, cuda_timing)]
    print(tabulate(timing_table, 
                  headers=['Method', 'Avg Time per Run (s)', 'Total Time (s)', 'Avg Time per Matrix (μs)'],
                  tablefmt='grid'))

    print("\n=== Accuracy Results ===")
    accuracy_table = [[r.method1, r.method2, f"{r.max_diff:.2e}", 
                      f"{r.avg_diff:.2e}", f"{r.success_rate:.2f}%"]
                     for r in accuracy_results]
    print(tabulate(accuracy_table,
                  headers=['Method 1', 'Method 2', 'Max Diff', 'Avg Diff', 'Success Rate'],
                  tablefmt='grid'))

    print("\n=== Special Cases Results ===")
    for case, results in special_case_results:
        print(f"\nTest case: {case.name}")
        print(f"Description: {case.description}")
        for r in results:
            if r.success_rate < 100:
                print(f"Failed comparison between {r.method1} and {r.method2}")
                print(f"Max difference: {r.max_diff:.2e}")

    # Generate plots
    plot_timing_comparison([numpy_timing, custom_timing, cuda_timing])
    print("\nTiming comparison plot saved as 'timing_comparison.png'")

if __name__ == "__main__":
    main()
