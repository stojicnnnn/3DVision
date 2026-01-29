#include "pipeline.hpp"
#include "pipeline_config.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>

using namespace industry_picking;

PipelineConfig loadConfig(const std::string& path) {
    PipelineConfig config;

    try {
        YAML::Node yaml = YAML::LoadFile(path);

        // Camera
        if (yaml["camera"]) {
            config.camera.width  = yaml["camera"]["width"].as<int>(1280);
            config.camera.height = yaml["camera"]["height"].as<int>(720);
        }

        // Depth
        if (yaml["depth"]) {
            config.depth.scale_to_meters  = yaml["depth"]["scale_to_meters"].as<float>(1000.0f);
            config.depth.clipping_max     = yaml["depth"]["clipping_max"].as<float>(1.5f);
            config.depth.bilateral_filter = yaml["depth"]["bilateral_filter"].as<bool>(false);
        }

        // Registration
        if (yaml["registration"]) {
            config.registration.voxel_size            = yaml["registration"]["voxel_size"].as<float>(0.001f);
            config.registration.ransac_max_iterations  = yaml["registration"]["ransac_max_iterations"].as<int>(100000);
            config.registration.icp_max_iterations     = yaml["registration"]["icp_max_iterations"].as<int>(200);
            config.registration.min_fitness            = yaml["registration"]["min_fitness"].as<float>(0.3f);
        }

        // Robot
        if (yaml["robot"]) {
            config.robot.ip              = yaml["robot"]["ip"].as<std::string>("192.168.1.184");
            config.robot.speed           = yaml["robot"]["speed"].as<int>(80);
            config.robot.approach_offset_z = yaml["robot"]["approach_offset_z"].as<float>(-0.101f);
        }

        // Segmentation
        if (yaml["segmentation"]) {
            config.segmentation.sam_server_url = yaml["segmentation"]["sam_server_url"].as<std::string>("");
            config.segmentation.sam_query      = yaml["segmentation"]["sam_query"].as<std::string>(
                "Segment the circular grey metallic caps,1 instance at a time, in order");
            config.segmentation.masks_input_dir = yaml["segmentation"]["masks_input_dir"].as<std::string>("");
            config.segmentation.apply_mask      = yaml["segmentation"]["apply_mask"].as<bool>(true);
        }

        // Paths
        config.reference_model_path = yaml["reference_model_path"].as<std::string>("");

        // Hardware & Dummy Data
        config.use_camera = yaml["use_camera"].as<bool>(true);
        config.use_robot  = yaml["use_robot"].as<bool>(true);
        if (yaml["dummy_data"]) {
            config.dummy_rgb_path   = yaml["dummy_data"]["rgb_path"].as<std::string>("");
            config.dummy_depth_path = yaml["dummy_data"]["depth_path"].as<std::string>("");
        }

        // Pipeline
        config.num_threads = yaml["num_threads"].as<int>(8);
        config.use_gpu     = yaml["use_gpu"].as<bool>(true);

        // Viz
        std::string viz = yaml["visualization"].as<std::string>("opengl");
        config.viz_backend = (viz == "none") ? VizBackend::NONE : VizBackend::OPENGL;

        // Camera extrinsics (4x4 matrix as flat array)
        if (yaml["camera_extrinsics"]) {
            auto ext = yaml["camera_extrinsics"];
            if (ext.size() == 16) {
                for (int i = 0; i < 16; ++i) {
                    config.camera_extrinsics(i / 4, i % 4) = ext[i].as<float>();
                }
            }
        }

        std::cout << "Config loaded from " << path << "\n";
    } catch (const YAML::Exception& e) {
        std::cerr << "Config error: " << e.what() << " — using defaults\n";
    }

    return config;
}

int main(int argc, char** argv) {
    std::cout << "=== Industry Picking — C++ GPU Pipeline ===\n\n";

    // Load config
    std::string config_path = "config/pipeline_config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    PipelineConfig config = loadConfig(config_path);

    // Run pipeline
    Pipeline pipeline(config);
    pipeline.run();

    return 0;
}
// Edit
// Edit
// Edit
// Edit
// Edit
// Edit
// Edit
