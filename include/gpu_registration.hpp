#pragma once

#include <Eigen/Core>
#include "registration.hpp"

namespace industry_picking {

class GPURegistration {
public:
    static RegistrationResult icpRefine(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initial_transform,
        float distance_threshold,
        int max_iterations  = 200
    );

    static bool isCudaAvailable();
};

}
