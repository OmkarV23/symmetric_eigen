#include <cuda_runtime.h>
#include <math_constants.h>

// CUDA kernel to compute eigenvalues for multiple 3x3 symmetric matrices using double precision
__global__ void compute_eigenvalues_cuda(double *A, double *eigenvalues, int num_matrices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_matrices) {
        int base_idx = idx * 9;
        double a11 = A[base_idx + 0];
        double a12 = A[base_idx + 1];
        double a13 = A[base_idx + 2];
        double a21 = A[base_idx + 3];
        double a22 = A[base_idx + 4];
        double a23 = A[base_idx + 5];
        double a31 = A[base_idx + 6];
        double a32 = A[base_idx + 7];
        double a33 = A[base_idx + 8];

        // Since the matrices are symmetric, you can average the symmetric elements
        a12 = (a12 + a21) / 2.0;
        a13 = (a13 + a31) / 2.0;
        a23 = (a23 + a32) / 2.0;

        double p1 = a12 * a12 + a13 * a13 + a23 * a23;
        double eig1, eig2, eig3;

        if (p1 == 0.0) {
            eig1 = a11;
            eig2 = a22;
            eig3 = a33;
        } else {
            double q = (a11 + a22 + a33) / 3.0;
            double p2 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + (a33 - q) * (a33 - q) + 2 * p1;
            double p = sqrt(p2 / 6.0);

            double b11 = (a11 - q) / p;
            double b12 = a12 / p;
            double b13 = a13 / p;
            double b22 = (a22 - q) / p;
            double b23 = a23 / p;
            double b33 = (a33 - q) / p;

            double detB = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b13 * b23) + b13 * (b12 * b23 - b13 * b22);
            double r = detB / 2.0;
            r = fmax(fmin(r, 1.0), -1.0);  // Clamp r to [-1, 1]

            // double phi;
            // if (r <= -1.0) {
            //     phi = CUDART_PI / 3.0;
            // } else if (r >= 1.0) {
            //     phi = 0.0;
            // } else {
            //     phi = acos(r) / 3.0;
            // }

            double phi = acos(r) / 3.0;

            eig1 = q + 2.0 * p * cos(phi);
            eig3 = q + 2.0 * p * cos(phi + (2.0 * CUDART_PI / 3.0));
            eig2 = 3.0 * q - eig1 - eig3;
        }

        eigenvalues[3 * idx + 0] = eig1;
        eigenvalues[3 * idx + 1] = eig2;
        eigenvalues[3 * idx + 2] = eig3;
    }
}

extern "C" {
    void launch_compute_eigenvalues(double *d_A, double *d_eigenvalues, int num_matrices) {
        int blockSize = 256;
        int gridSize = (num_matrices + blockSize - 1) / blockSize;
        compute_eigenvalues_cuda<<<gridSize, blockSize>>>(d_A, d_eigenvalues, num_matrices);
    }
}