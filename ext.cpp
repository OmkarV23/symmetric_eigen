#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath>

namespace py = pybind11;

extern "C" {
    void launch_compute_eigenvalues(double *d_A, double *d_eigenvalues, int num_matrices);
}

void compute_eigenvalues(py::array_t<double, py::array::c_style | py::array::forcecast> input_matrices_flat,
                         py::array_t<double> output_eigenvalues) {
    auto buf_in = input_matrices_flat.request();
    auto buf_out = output_eigenvalues.request();

    int num_matrices = buf_in.shape[0];
    if (buf_in.shape[1] != 9) {
        throw std::runtime_error("Input matrices should be of shape (num_matrices, 9)");
    }

    double *A = static_cast<double*>(buf_in.ptr);
    double *eigenvalues = static_cast<double*>(buf_out.ptr);

    // Allocate memory on device
    double *d_A, *d_eigenvalues;
    cudaMalloc((void**)&d_A, num_matrices * 9 * sizeof(double));
    cudaMalloc((void**)&d_eigenvalues, num_matrices * 3 * sizeof(double));

    // Copy input matrices to device
    cudaMemcpy(d_A, A, num_matrices * 9 * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    launch_compute_eigenvalues(d_A, d_eigenvalues, num_matrices);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + cudaGetErrorString(err));
    }

    // Synchronize device to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Check for any errors during kernel execution
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution error: ") + cudaGetErrorString(err));
    }

    // Copy results back
    cudaMemcpy(eigenvalues, d_eigenvalues, num_matrices * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_eigenvalues);
}

PYBIND11_MODULE(eigenvalues_cuda, m) {
    m.def("compute_eigenvalues", &compute_eigenvalues, "Compute eigenvalues for multiple 3x3 symmetric matrices");
}