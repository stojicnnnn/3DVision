#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {


void launchDeproject(
    const float* depth,
    const unsigned char* rgb,
    float* out_points,
    int* out_count,
    int width, int height,
    float fx, float fy, float cx, float cy,
    float max_depth
);

}
}
