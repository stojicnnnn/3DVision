#pragma once

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace industry_picking {
namespace cuda {


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

}
}
