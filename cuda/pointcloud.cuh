#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {

/**
 * @brief GPU RGBD → point cloud deprojection.
 *
 * Each thread processes one pixel, converting (u, v, depth) → (X, Y, Z, R, G, B).
 * Uses atomic counter for output compaction (skip zero-depth pixels).
 *
 * @param depth         Device ptr: float depth in meters (HxW)
 * @param rgb           Device ptr: RGB image (HxWx3, uint8)
 * @param out_points    Device ptr: output Nx6 (XYZRGB), pre-allocated to HxW*6
 * @param out_count     Device ptr: atomic counter for valid points
 * @param width, height Image dimensions
 * @param fx, fy        Focal lengths
 * @param cx, cy        Principal point
 * @param max_depth     Maximum valid depth (meters)
 */
void launchDeproject(
    const float* depth,
    const unsigned char* rgb,
    float* out_points,
    int* out_count,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float max_depth
);

}  // namespace cuda
}  // namespace industry_picking
