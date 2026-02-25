#include "camera.hpp"
#include <iostream>

namespace industry_picking {

RealSenseCamera::RealSenseCamera(int width, int height)
    : width_(width), height_(height) {
    std::cout << "RealSense camera created (" << width << "x" << height << ")\n";
}

RealSenseCamera::~RealSenseCamera() {
    if (connected_) disconnect();
}

bool RealSenseCamera::connect() {
    try {
        config_.enable_stream(RS2_STREAM_COLOR, width_, height_, RS2_FORMAT_BGR8, 30);
        config_.enable_stream(RS2_STREAM_DEPTH, width_, height_, RS2_FORMAT_Z16, 30);

        profile_ = pipeline_.start(config_);

        // Get depth scale
        auto depth_sensor = profile_.get_device().first<rs2::depth_sensor>();
        depth_scale_ = depth_sensor.get_depth_scale();

        connected_ = true;
        std::cout << "RealSense connected. Depth scale: " << depth_scale_ << "\n";

        // Allow auto-exposure to settle
        for (int i = 0; i < 30; ++i) pipeline_.wait_for_frames();

        return true;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << "\n";
        return false;
    }
}

void RealSenseCamera::disconnect() {
    if (connected_) {
        pipeline_.stop();
        connected_ = false;
        std::cout << "RealSense disconnected.\n";
    }
}

bool RealSenseCamera::capture(cv::Mat& rgb, cv::Mat& depth) {
    if (!connected_) {
        std::cerr << "Camera not connected.\n";
        return false;
    }

    try {
        // Align depth to color
        rs2::align align(RS2_STREAM_COLOR);
        auto frames = pipeline_.wait_for_frames();
        auto aligned = align.process(frames);

        auto color_frame = aligned.get_color_frame();
        auto depth_frame = aligned.get_depth_frame();

        if (!color_frame || !depth_frame) return false;

        // Wrap as cv::Mat (no copy)
        rgb = cv::Mat(
            cv::Size(color_frame.get_width(), color_frame.get_height()),
            CV_8UC3,
            const_cast<void*>(color_frame.get_data())
        ).clone();

        depth = cv::Mat(
            cv::Size(depth_frame.get_width(), depth_frame.get_height()),
            CV_16UC1,
            const_cast<void*>(depth_frame.get_data())
        ).clone();

        return true;
    } catch (const rs2::error& e) {
        std::cerr << "Capture error: " << e.what() << "\n";
        return false;
    }
}

Eigen::Matrix3f RealSenseCamera::getIntrinsics() const {
    auto stream = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intr = stream.get_intrinsics();

    Eigen::Matrix3f K;
    K << intr.fx, 0.0f,    intr.ppx,
         0.0f,    intr.fy, intr.ppy,
         0.0f,    0.0f,    1.0f;
    return K;
}

float RealSenseCamera::getDepthScale() const {
    return depth_scale_;
}

} 
