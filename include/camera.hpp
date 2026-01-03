#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <librealsense2/rs.hpp>
#include <string>

namespace industry_picking {

/**
 * @brief Wrapper around Intel RealSense camera using librealsense2 C++ API.
 */
class RealSenseCamera {
public:
    RealSenseCamera(int width = 1280, int height = 720);
    ~RealSenseCamera();

    // Non-copyable
    RealSenseCamera(const RealSenseCamera&) = delete;
    RealSenseCamera& operator=(const RealSenseCamera&) = delete;

    /** @brief Start the camera pipeline. */
    bool connect();

    /** @brief Stop the camera pipeline. */
    void disconnect();

    /**
     * @brief Capture an aligned RGB + depth frame pair.
     * @param[out] rgb   BGR image (CV_8UC3)
     * @param[out] depth Depth image (CV_16UC1), values in raw sensor units
     * @return true on success
     */
    bool capture(cv::Mat& rgb, cv::Mat& depth);

    /**
     * @brief Get the 3x3 camera intrinsic matrix K.
     * @return Intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
     */
    Eigen::Matrix3f getIntrinsics() const;

    /** @brief Get the depth scale (units per meter). */
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

}  // namespace industry_picking
