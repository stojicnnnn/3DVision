#include "depth_processing.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>

namespace industry_picking {
namespace cuda {

// ─── Depth Preprocessing Kernel ───

__global__ void depthPreprocessKernel(
    const unsigned short* __restrict__ raw_depth,
    const unsigned char*  __restrict__ mask,
    float* __restrict__               out_depth,
    int width, int height,
    float inv_scale,
    bool apply_mask
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned short raw = raw_depth[idx];

    // Apply mask: zero out masked pixels
    if (apply_mask && mask != nullptr && mask[idx] == 0) {
        out_depth[idx] = 0.0f;
        return;
    }

    // Scale to meters
    out_depth[idx] = static_cast<float>(raw) * inv_scale;
}

void launchDepthPreprocess(
    const unsigned short* raw_depth,
    const unsigned char* mask,
    float* out_depth,
    int width, int height,
    float scale,
    bool apply_mask
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    float inv_scale = 1.0f / scale;

    depthPreprocessKernel<<<grid, block>>>(
        raw_depth, mask, out_depth,
        width, height, inv_scale, apply_mask
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA depth preprocess error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// ─── Bilateral Filter Kernel ───

__global__ void bilateralFilterKernel(
    const float* __restrict__ in_depth,
    float* __restrict__       out_depth,
    int width, int height,
    float sigma_spatial,
    float sigma_range
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float center = in_depth[idx];
    if (center <= 0.0f) {
        out_depth[idx] = 0.0f;
        return;
    }

    int radius = static_cast<int>(2.0f * sigma_spatial);
    float inv_spatial2 = -0.5f / (sigma_spatial * sigma_spatial);
    float inv_range2   = -0.5f / (sigma_range * sigma_range);

    float sum_weight = 0.0f;
    float sum_value  = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            float neighbor = in_depth[ny * width + nx];
            if (neighbor <= 0.0f) continue;

            float spatial_dist2 = static_cast<float>(dx * dx + dy * dy);
            float range_diff    = neighbor - center;
            float range_dist2   = range_diff * range_diff;

            float weight = expf(spatial_dist2 * inv_spatial2 + range_dist2 * inv_range2);
            sum_weight += weight;
            sum_value  += weight * neighbor;
        }
    }

    out_depth[idx] = (sum_weight > 0.0f) ? (sum_value / sum_weight) : center;
}

void launchBilateralFilter(
    const float* in_depth,
    float* out_depth,
    int width, int height,
    float sigma_spatial,
    float sigma_range
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    bilateralFilterKernel<<<grid, block>>>(
        in_depth, out_depth, width, height, sigma_spatial, sigma_range
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA bilateral filter error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace industry_picking

#endif  // CUDA_AVAILABLE
