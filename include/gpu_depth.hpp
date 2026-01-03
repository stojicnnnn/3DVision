#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "registration.hpp"

namespace industry_picking {

/**
 * @brief GPU-accelerated depth preprocessing.
 * Falls back to CPU when CUDA is not available.
 */
class GPUDepth {
public:
    /**
     * @brief Scale raw depth to meters and apply a binary mask on the GPU.
     * @param raw_depth   Raw depth image (CV_16UC1)
     * @param mask        Binary mask (CV_8UC1, 255=foreground), empty to skip masking
     * @param scale       Depth scale (raw units per meter, e.g. 1000.0 for mm)
     * @return Scaled float depth in meters (CV_32FC1)
     */
    static cv::Mat preprocess(const cv::Mat& raw_depth, const cv::Mat& mask, float scale);

    /** @brief Returns true if CUDA is available at runtime. */
    static bool isCudaAvailable();
};

/**
 * @brief GPU-accelerated RGBD → point cloud deprojection.
 * Falls back to CPU when CUDA is not available.
 */
class GPUPointCloud {
public:
    /**
     * @brief Deproject a depth image into a 3D point cloud using camera intrinsics.
     * @param depth   Scaled depth image in meters (CV_32FC1)
     * @param rgb     Color image (CV_8UC3, BGR)
     * @param fx, fy  Focal lengths
     * @param cx, cy  Principal point
     * @return Point cloud with colors
     */
    static PointCloud generate(
        const cv::Mat& depth,
        const cv::Mat& rgb,
        float fx, float fy, float cx, float cy
    );
};

}  // namespace industry_picking
