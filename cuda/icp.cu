#include "icp.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace industry_picking {
namespace cuda {

// ─── Find Correspondences Kernel ───
// Each thread handles one source point: brute-force NN search in target.

__global__ void findCorrespondencesKernel(
    const float* __restrict__ source,       // Nx3
    const float* __restrict__ target,       // Mx3
    int*   __restrict__       correspondences,  // N
    float* __restrict__       distances,        // N
    int n_source, int n_target,
    const float* __restrict__ transform,    // 4x4 row-major
    float max_distance_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_source) return;

    // Transform source point
    float sx = source[i * 3 + 0];
    float sy = source[i * 3 + 1];
    float sz = source[i * 3 + 2];

    float px = transform[0] * sx + transform[1] * sy + transform[2]  * sz + transform[3];
    float py = transform[4] * sx + transform[5] * sy + transform[6]  * sz + transform[7];
    float pz = transform[8] * sx + transform[9] * sy + transform[10] * sz + transform[11];

    // Brute-force nearest neighbor in target
    float best_dist2 = FLT_MAX;
    int best_idx = -1;

    for (int j = 0; j < n_target; ++j) {
        float dx = px - target[j * 3 + 0];
        float dy = py - target[j * 3 + 1];
        float dz = pz - target[j * 3 + 2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_idx = j;
        }
    }

    if (best_dist2 < max_distance_sq && best_idx >= 0) {
        correspondences[i] = best_idx;
        distances[i] = best_dist2;
    } else {
        correspondences[i] = -1;
        distances[i] = FLT_MAX;
    }
}

void launchFindCorrespondences(
    const float* source,
    const float* target,
    int* correspondences,
    float* distances,
    int n_source,
    int n_target,
    const float* transform,
    float max_distance
) {
    int block_size = 256;
    int grid_size = (n_source + block_size - 1) / block_size;

    float max_dist_sq = max_distance * max_distance;

    findCorrespondencesKernel<<<grid_size, block_size>>>(
        source, target,
        correspondences, distances,
        n_source, n_target,
        transform, max_dist_sq
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA find correspondences error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// ─── Build Linear System Kernel ───
// Accumulates JᵀJ (6x6) and Jᵀr (6x1) using atomicAdd.

__global__ void buildLinearSystemKernel(
    const float* __restrict__ source,          // transformed Nx3
    const float* __restrict__ target,          // Mx3
    const float* __restrict__ target_normals,  // Mx3
    const int*   __restrict__ correspondences, // N
    const float* __restrict__ distances,       // N
    float* __restrict__       JtJ,             // 36 floats (6x6)
    float* __restrict__       Jtr,             // 6 floats
    int n_source,
    float max_distance_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_source) return;

    int corr_idx = correspondences[i];
    if (corr_idx < 0 || distances[i] >= max_distance_sq) return;

    float px = source[i * 3 + 0];
    float py = source[i * 3 + 1];
    float pz = source[i * 3 + 2];

    float qx = target[corr_idx * 3 + 0];
    float qy = target[corr_idx * 3 + 1];
    float qz = target[corr_idx * 3 + 2];

    float nx = target_normals[corr_idx * 3 + 0];
    float ny = target_normals[corr_idx * 3 + 1];
    float nz = target_normals[corr_idx * 3 + 2];

    // Cross product: p × n
    float cx = py * nz - pz * ny;
    float cy = pz * nx - px * nz;
    float cz = px * ny - py * nx;

    // Jacobian row: [cx, cy, cz, nx, ny, nz]
    float J[6] = {cx, cy, cz, nx, ny, nz};

    // Residual: (p - q) · n
    float residual = (px - qx) * nx + (py - qy) * ny + (pz - qz) * nz;

    // Accumulate JᵀJ and Jᵀr
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            atomicAdd(&JtJ[r * 6 + c], J[r] * J[c]);
        }
        atomicAdd(&Jtr[r], J[r] * residual);
    }
}

void launchBuildLinearSystem(
    const float* source,
    const float* target,
    const float* target_normals,
    const int* correspondences,
    const float* distances,
    float* JtJ,
    float* Jtr,
    int n_source,
    float max_distance
) {
    // Zero-initialize JtJ and Jtr
    cudaMemset(JtJ, 0, 36 * sizeof(float));
    cudaMemset(Jtr, 0, 6 * sizeof(float));

    int block_size = 256;
    int grid_size = (n_source + block_size - 1) / block_size;

    float max_dist_sq = max_distance * max_distance;

    buildLinearSystemKernel<<<grid_size, block_size>>>(
        source, target, target_normals,
        correspondences, distances,
        JtJ, Jtr,
        n_source, max_dist_sq
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA build linear system error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace industry_picking

#endif  // CUDA_AVAILABLE
