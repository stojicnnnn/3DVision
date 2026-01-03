#pragma once

#include <string>
#include <array>
#include <Eigen/Core>

namespace industry_picking {

enum class VizBackend { NONE, OPENGL };

struct CameraConfig {
    int width  = 1280;
    int height = 720;
    std::string ip;  // Empty for USB cameras
};

struct DepthConfig {
    float scale_to_meters = 1000.0f;  // Raw depth units per meter
    float clipping_min    = 0.1f;     // Min depth in meters
    float clipping_max    = 1.5f;     // Max depth in meters
    bool  bilateral_filter = false;
};

struct RegistrationConfig {
    float voxel_size                  = 0.001f;
    int   ransac_max_iterations       = 100000;
    float ransac_confidence           = 0.999f;
    float icp_distance_factor         = 0.4f;
    int   icp_max_iterations          = 200;
    float min_fitness                  = 0.3f;
    bool  use_point_to_plane          = true;
};

struct RobotConfig {
    std::string ip        = "192.168.1.184";
    int         speed     = 80;
    float       approach_offset_z = -0.101f;  // meters
};

struct SegmentationConfig {
    std::string sam_server_url;
    std::string sam_query = "Segment the circular grey metallic caps,1 instance at a time, in order";
    std::string masks_input_dir;
    bool        apply_mask = true;
};

struct PipelineConfig {
    CameraConfig       camera;
    DepthConfig        depth;
    RegistrationConfig registration;
    RobotConfig        robot;
    SegmentationConfig segmentation;

    std::string reference_model_path;

    // Hardware flags
    bool use_camera = true;
    bool use_robot  = true;

    // Dummy data (if use_camera = false)
    std::string dummy_rgb_path;
    std::string dummy_depth_path;

    // Concurrency
    int  num_threads = 8;
    bool use_gpu     = true;

    // Visualization
    VizBackend viz_backend = VizBackend::OPENGL;

    // Camera extrinsics (4x4, row-major)
    Eigen::Matrix4f camera_extrinsics = Eigen::Matrix4f::Identity();
};

}  // namespace industry_picking
