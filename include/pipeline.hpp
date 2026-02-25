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

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    void run();

private:
    std::optional<Eigen::Matrix4f> processInstance(
        const cv::Mat& mask,
        const cv::Mat& depth,
        const cv::Mat& rgb,
        const Eigen::Matrix3f& intrinsics,
        const PointCloud& ref_cloud,
        const FPFHFeatures& ref_features,
        int instance_id
    );

    std::vector<Eigen::Matrix4f> filterDuplicates(
        const std::vector<Eigen::Matrix4f>& waypoints,
        float min_distance
    );

    PipelineConfig config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<GLViewer>   viewer_;
};

}
