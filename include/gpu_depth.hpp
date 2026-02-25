#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "registration.hpp"

namespace industry_picking {

class GPUDepth {
public:
    static cv::Mat preprocess(const cv::Mat& raw_depth, const cv::Mat& mask, float scale);
    static bool isCudaAvailable();
};

class GPUPointCloud {
public:
    static PointCloud generate(
        const cv::Mat& depth,
        const cv::Mat& rgb,
        float fx, float fy, float cx, float cy
    );
};

}
