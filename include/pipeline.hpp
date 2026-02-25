#pragma once

#include "pipeline_config.hpp"
#include "camera.hpp"
#include "segmentation.hpp"
#include "registration.hpp"
#include "robot.hpp"
#include "gl_viewer.hpp"
#include "thread_pool.hpp"
#include <memory>
#include <vector>

namespace industry_picking {

/**
 * @brief Main pipeline orchestrator.
 *
 * Wires together: capture → segmentation → point cloud → registration → pick.
 * Supports GPU acceleration and concurrent mask processing.
 */
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    /** @brief Run the full picking pipeline. */
    void run();

private:
    /**
     * @brief Process a single mask instance: depth → point cloud → registration → pose.
     * @param mask              Binary mask (CV_8UC1)
     * @param depth             Raw depth image (CV_16UC1)
     * @param rgb               Color image (CV_8UC3)
     * @param intrinsics        Camera intrinsic matrix
     * @param ref_cloud         Reference model point cloud (downsampled)
     * @param ref_features      Reference model FPFH features
     * @param instance_id       Index for logging
     * @return 4x4 world pose or std::nullopt on failure
     */
    std::optional<Eigen::Matrix4f> processInstance(
        const cv::Mat& mask,
        const cv::Mat& depth,
        const cv::Mat& rgb,
        const Eigen::Matrix3f& intrinsics,
        const PointCloud& ref_cloud,
        const FPFHFeatures& ref_features,
        int instance_id
    );

    /**
     * @brief Filter duplicate waypoints by proximity.
     */
    std::vector<Eigen::Matrix4f> filterDuplicates(
        const std::vector<Eigen::Matrix4f>& waypoints,
        float min_distance
    );

    PipelineConfig config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<GLViewer>   viewer_;
};

}  // namespace industry_picking
