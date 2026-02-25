#include "icp.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

namespace industry_picking {
namespace cuda {

// Brute-force O(N·M) nearest-neighbor search. Fine for small clouds (couple of k pts);
// for large clouds, swap in a GPU k-d tree cuspatial or gpu-flann

__global__ void findCorrespondencesKernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    int*   __restrict__       correspondences,
    float* __restrict__       distances,
    int n_source, int n_target,
    const float* __restrict__ transform,
    float max_distance_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_source) return;

    float sx = source[i * 3 + 0];
    float sy = source[i * 3 + 1];
    float sz = source[i * 3 + 2];

    float px = transform[0]*sx + transform[1]*sy + transform[2] *sz + transform[3];
    float py = transform[4]*sx + transform[5]*sy + transform[6] *sz + transform[7];
    float pz = transform[8]*sx + transform[9]*sy + transform[10]*sz + transform[11];

    float best_dist2 = FLT_MAX;
    int   best_idx   = -1;

    for (int j = 0; j < n_target; ++j) {
        float dx = px - __ldg(&target[j * 3 + 0]);
        float dy = py - __ldg(&target[j * 3 + 1]);
        float dz = pz - __ldg(&target[j * 3 + 2]);
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_idx   = j;
        }
    }

    if (best_idx >= 0 && best_dist2 < max_distance_sq) {
        correspondences[i] = best_idx;
        distances[i]       = best_dist2;
    } else {
        correspondences[i] = -1;
        distances[i]       = FLT_MAX;
    }
}

void launchFindCorrespondences(
    const float* source,
    const float* target,
    int*         correspondences,
    float*       distances,
    int n_source, int n_target,
    const float* transform,
    float max_distance
) {
    constexpr int kBlock = 256;
    int grid = (n_source + kBlock - 1) / kBlock;

    findCorrespondencesKernel<<<grid, kBlock>>>(
        source, target,
        correspondences, distances,
        n_source, n_target,
        transform, max_distance * max_distance
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[FindCorrespondences] CUDA error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}


__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void buildLinearSystemKernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    const float* __restrict__ target_normals,
    const int*   __restrict__ correspondences,
    const float* __restrict__ distances,
    float* __restrict__       JtJ,
    float* __restrict__       Jtr,
    int   n_source,
    float max_distance_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float J[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float r    = 0.f;

    if (i < n_source) {
        int corr = correspondences[i];
        if (corr >= 0 && distances[i] < max_distance_sq) {
            float px = source[i * 3 + 0];
            float py = source[i * 3 + 1];
            float pz = source[i * 3 + 2];

            float qx = target[corr * 3 + 0];
            float qy = target[corr * 3 + 1];
            float qz = target[corr * 3 + 2];

            float nx = target_normals[corr * 3 + 0];
            float ny = target_normals[corr * 3 + 1];
            float nz = target_normals[corr * 3 + 2];

            // Jacobian: [p × n | n]
            J[0] = py*nz - pz*ny;
            J[1] = pz*nx - px*nz;
            J[2] = px*ny - py*nx;
            J[3] = nx;  J[4] = ny;  J[5] = nz;

            r = (px - qx)*nx + (py - qy)*ny + (pz - qz)*nz;
        }
    }

    for (int row = 0; row < 6; ++row) {
        float jr = warpReduceSum(J[row] * r);
        for (int col = 0; col < 6; ++col) {
            float jj = warpReduceSum(J[row] * J[col]);
            if ((threadIdx.x & (warpSize - 1)) == 0) {
                atomicAdd(&JtJ[row * 6 + col], jj);
            }
        }
        if ((threadIdx.x & (warpSize - 1)) == 0)
            atomicAdd(&Jtr[row], jr);
    }
}

void launchBuildLinearSystem(
    const float* source,
    const float* target,
    const float* target_normals,
    const int*   correspondences,
    const float* distances,
    float*       JtJ,
    float*       Jtr,
    int   n_source,
    float max_distance
) {
    cudaMemset(JtJ, 0, 36 * sizeof(float));
    cudaMemset(Jtr, 0,  6 * sizeof(float));

    constexpr int kBlock = 256;
    int grid = (n_source + kBlock - 1) / kBlock;

    buildLinearSystemKernel<<<grid, kBlock>>>(
        source, target, target_normals,
        correspondences, distances,
        JtJ, Jtr,
        n_source, max_distance * max_distance
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BuildLinearSystem] CUDA error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

}
}

#endif
