#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {

/**
 * @brief GPU-accelerated ICP: find nearest neighbors in parallel.
 *
 * @param source        Device ptr: source points (Nx3)
 * @param target        Device ptr: target points (Mx3)
 * @param target_normals Device ptr: target normals (Mx3), or nullptr for point-to-point
 * @param correspondences Device ptr: output correspondence indices (N)
 * @param distances     Device ptr: output distances (N)
 * @param n_source      Number of source points
 * @param n_target      Number of target points
 * @param transform     Device ptr: current 4x4 transform (row-major)
 * @param max_distance  Maximum correspondence distance
 */
void launchFindCorrespondences(
    const float* source,
    const float* target,
    int* correspondences,
    float* distances,
    int n_source,
    int n_target,
    const float* transform,
    float max_distance
);

/**
 * @brief Build the 6x6 linear system (JᵀJ, Jᵀr) for point-to-plane ICP.
 *
 * @param source        Transformed source points (Nx3)
 * @param target        Target points (Mx3)
 * @param target_normals Target normals (Mx3)
 * @param correspondences Correspondence indices (N)
 * @param distances     Correspondence distances (N)
 * @param JtJ           Output: 6x6 matrix (36 floats), accumulated via atomicAdd
 * @param Jtr           Output: 6x1 vector (6 floats), accumulated via atomicAdd
 * @param n_source      Number of source points
 * @param max_distance  Distance threshold for inliers
 */
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
);

}  // namespace cuda
}  // namespace industry_picking
