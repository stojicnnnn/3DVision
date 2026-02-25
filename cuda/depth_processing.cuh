#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {


void launchDepthPreprocess(
    const unsigned short* raw_depth,
    const unsigned char* mask,
    float* out_depth,
    int width, int height,
    float scale,
    bool apply_mask
);

void launchBilateralFilter(
    const float* in_depth,
    float* out_depth,
    int width, int height,
    float sigma_spatial,
    float sigma_range
);

}
}
