#include "pointcloud.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>

namespace industry_picking {
namespace cuda {

__global__ void deprojectKernel(
    const float* __restrict__         depth,
    const unsigned char* __restrict__ rgb,
    float* __restrict__               out_points,
    int* __restrict__                 out_count,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float max_depth
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    int pixel_idx = v * width + u;
    float z = depth[pixel_idx];

    // Skip invalid depth
    if (z <= 0.0f || z > max_depth) return;

    // Deprojection
    float x = (static_cast<float>(u) - cx) * z / fx;
    float y = (static_cast<float>(v) - cy) * z / fy;

    // Get RGB (BGR format from OpenCV)
    int rgb_idx = pixel_idx * 3;
    float r = static_cast<float>(rgb[rgb_idx + 2]) / 255.0f;
    float g = static_cast<float>(rgb[rgb_idx + 1]) / 255.0f;
    float b = static_cast<float>(rgb[rgb_idx + 0]) / 255.0f;

    // Atomic insert into output array
    int out_idx = atomicAdd(out_count, 1);
    int base = out_idx * 6;
    out_points[base + 0] = x;
    out_points[base + 1] = y;
    out_points[base + 2] = z;
    out_points[base + 3] = r;
    out_points[base + 4] = g;
    out_points[base + 5] = b;
}

void launchDeproject(
    const float* depth,
    const unsigned char* rgb,
    float* out_points,
    int* out_count,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float max_depth
) {
    // Reset counter
    cudaMemset(out_count, 0, sizeof(int));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    deprojectKernel<<<grid, block>>>(
        depth, rgb, out_points, out_count,
        width, height, fx, fy, cx, cy, max_depth
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA deproject error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace industry_picking

#endif  // CUDA_AVAILABLE
