#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {

/**
 * @brief GPU depth preprocessing: scale raw depth to meters + apply mask.
 *
 * @param raw_depth     Device ptr: raw 16-bit depth (HxW)
 * @param mask          Device ptr: binary mask (HxW, 255=keep), or nullptr
 * @param out_depth     Device ptr: output float depth in meters (HxW)
 * @param width         Image width
 * @param height        Image height
 * @param scale         Depth scale (raw units per meter)
 * @param apply_mask    Whether to apply the mask
 */
void launchDepthPreprocess(
    const unsigned short* raw_depth,
    const unsigned char* mask,
    float* out_depth,
    int width, int height,
    float scale,
    bool apply_mask
);

/**
 * @brief GPU bilateral filter on depth image.
 */
void launchBilateralFilter(
    const float* in_depth,
    float* out_depth,
    int width, int height,
    float sigma_spatial,
    float sigma_range
);

}  // namespace cuda
}  // namespace industry_picking
