#include "depth_processing.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cstdio>

namespace industry_picking {
namespace cuda {

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

    if (apply_mask && mask != nullptr && mask[idx] == 0) {
        out_depth[idx] = 0.0f;
        return;
    }

    out_depth[idx] = static_cast<float>(raw_depth[idx]) * inv_scale;
}

void launchDepthPreprocess(
    const unsigned short* raw_depth,
    const unsigned char*  mask,
    float*                out_depth,
    int width, int height,
    float scale,
    bool apply_mask
) {
    dim3 block(32, 8);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    depthPreprocessKernel<<<grid, block>>>(
        raw_depth, mask, out_depth,
        width, height, 1.0f / scale, apply_mask
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[DepthPreprocess] CUDA error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

#define BF_TILE_W 16
#define BF_TILE_H 16
#define BF_MAX_RADIUS 5
#define BF_SMEM_W (BF_TILE_W + 2 * BF_MAX_RADIUS)
#define BF_SMEM_H (BF_TILE_H + 2 * BF_MAX_RADIUS)

__global__ void bilateralFilterKernel(
    const float* __restrict__ in_depth,
    float* __restrict__       out_depth,
    int width, int height,
    int   radius,
    float inv_spatial2,
    float inv_range2
) {
    __shared__ float smem[BF_SMEM_H][BF_SMEM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx0 = blockIdx.x * BF_TILE_W - radius;
    int gy0 = blockIdx.y * BF_TILE_H - radius;

    int smem_w = BF_TILE_W + 2 * radius;
    int smem_h = BF_TILE_H + 2 * radius;

    for (int sy = ty; sy < smem_h; sy += BF_TILE_H) {
        for (int sx = tx; sx < smem_w; sx += BF_TILE_W) {
            int gx = gx0 + sx;
            int gy = gy0 + sy;
            float val = 0.0f;
            if (gx >= 0 && gx < width && gy >= 0 && gy < height)
                val = in_depth[gy * width + gx];
            smem[sy][sx] = val;
        }
    }
    __syncthreads();

    int x = blockIdx.x * BF_TILE_W + tx;
    int y = blockIdx.y * BF_TILE_H + ty;
    if (x >= width || y >= height) return;

    int scx = tx + radius;
    int scy = ty + radius;

    float center = smem[scy][scx];
    if (center <= 0.0f) {
        out_depth[y * width + x] = 0.0f;
        return;
    }

    float sum_w = 0.0f, sum_v = 0.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float nb = smem[scy + dy][scx + dx];
            if (nb <= 0.0f) continue;

            float rd = nb - center;
            float w  = expf(static_cast<float>(dx*dx + dy*dy) * inv_spatial2
                            + rd * rd * inv_range2);
            sum_w += w;
            sum_v += w * nb;
        }
    }

    out_depth[y * width + x] = (sum_w > 0.0f) ? (sum_v / sum_w) : center;
}

void launchBilateralFilter(
    const float* in_depth,
    float*       out_depth,
    int width, int height,
    float sigma_spatial,
    float sigma_range
) {
    int radius = static_cast<int>(2.0f * sigma_spatial + 0.5f);
    if (radius > BF_MAX_RADIUS) {
        fprintf(stderr, "[BilateralFilter] sigma_spatial too large (radius %d > %d); clamping\n",
                radius, BF_MAX_RADIUS);
        radius = BF_MAX_RADIUS;
    }

    float inv_spatial2 = -0.5f / (sigma_spatial * sigma_spatial);
    float inv_range2   = -0.5f / (sigma_range   * sigma_range);

    dim3 block(BF_TILE_W, BF_TILE_H);
    dim3 grid((width  + BF_TILE_W - 1) / BF_TILE_W,
              (height + BF_TILE_H - 1) / BF_TILE_H);

    bilateralFilterKernel<<<grid, block>>>(
        in_depth, out_depth, width, height,
        radius, inv_spatial2, inv_range2
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BilateralFilter] CUDA error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

}
}

#endif
