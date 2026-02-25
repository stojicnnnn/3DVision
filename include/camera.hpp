#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <librealsense2/rs.hpp>
#include <string>

namespace industry_picking {

class RealSenseCamera {
public:
    RealSenseCamera(int width = 1280, int height = 720);
    ~RealSenseCamera();

    RealSenseCamera(const RealSenseCamera&) = delete;
    RealSenseCamera& operator=(const RealSenseCamera&) = delete;

    bool connect();
    void disconnect();
    bool capture(cv::Mat& rgb, cv::Mat& depth);
    Eigen::Matrix3f getIntrinsics() const;
    float getDepthScale() const;
    bool isConnected() const { return connected_; }

private:
    int width_, height_;
    bool connected_ = false;

    rs2::pipeline   pipeline_;
    rs2::config     config_;
    rs2::pipeline_profile profile_;
    float depth_scale_ = 0.001f;
};

}
