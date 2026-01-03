#pragma once

#include <Eigen/Core>
#include "registration.hpp"

namespace industry_picking {

/**
 * @brief GPU-accelerated ICP registration.
 * Falls back to CPU Registration::icpRefine when CUDA is not available.
 */
class GPURegistration {
public:
    /**
     * @brief GPU-accelerated ICP point-to-plane refinement.
     */
    static RegistrationResult icpRefine(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initial_transform,
        float distance_threshold,
        int max_iterations  = 200
    );

    /** @brief Returns true if CUDA is available at runtime. */
    static bool isCudaAvailable();
};

}  // namespace industry_picking
