#include "pointcloud.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>

namespace industry_picking {
namespace cuda {


__global__ void deprojectKernel(
    const float*          __restrict__ depth,
    const unsigned char*  __restrict__ rgb,
    float*                __restrict__ out_points,
    int*                  __restrict__ out_count,
    int width, int height,
    float fx_inv, float fy_inv,
    float cx, float cy,
    float max_depth,
    int   max_points 
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    int pixel_idx = v * width + u;
    float z = __ldg(&depth[pixel_idx]);
    if (z <= 0.0f || z > max_depth) return;

    float x = (static_cast<float>(u) - cx) * z * fx_inv;
    float y = (static_cast<float>(v) - cy) * z * fy_inv;

    int rgb_base = pixel_idx * 3;
    float r = static_cast<float>(__ldg(&rgb[rgb_base + 2])) * (1.0f / 255.0f);
    float g = static_cast<float>(__ldg(&rgb[rgb_base + 1])) * (1.0f / 255.0f);
    float b = static_cast<float>(__ldg(&rgb[rgb_base + 0])) * (1.0f / 255.0f);

    int out_idx = atomicAdd(out_count, 1);
    if (out_idx >= max_points) {
        atomicSub(out_count, 1);
        return;
    }

    int base = out_idx * 6;
    out_points[base + 0] = x;
    out_points[base + 1] = y;
    out_points[base + 2] = z;
    out_points[base + 3] = r;
    out_points[base + 4] = g;
    out_points[base + 5] = b;
}

void launchDeproject(
    const float*         depth,
    const unsigned char* rgb,
    float*               out_points,
    int*                 out_count,
    int width, int height,
    float fx, float fy,
    float cx, float cy,
    float max_depth
) {
    cudaMemset(out_count, 0, sizeof(int));

    dim3 block(32, 8);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    deprojectKernel<<<grid, block>>>(
        depth, rgb,
        out_points, out_count,
        width, height,
        1.0f / fx, 1.0f / fy,
        cx, cy,
        max_depth,
        width * height
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[Deproject] CUDA error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

}
}

#endif
