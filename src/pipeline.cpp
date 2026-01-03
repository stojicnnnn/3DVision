#include "pipeline.hpp"
#include "gpu_depth.hpp"

#include "gpu_registration.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <optional>
#include <chrono>

namespace industry_picking {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config)
    , thread_pool_(std::make_unique<ThreadPool>(config.num_threads))
{
    std::cout << "Pipeline created (threads=" << config.num_threads
              << ", gpu=" << (config.use_gpu ? "on" : "off") << ")\n";
}

Pipeline::~Pipeline() = default;

// ─── Process a single segmentation instance ───

std::optional<Eigen::Matrix4f> Pipeline::processInstance(
    const cv::Mat& mask,
    const cv::Mat& depth,
    const cv::Mat& rgb,
    const Eigen::Matrix3f& intrinsics,
    const PointCloud& ref_cloud,
    const FPFHFeatures& ref_features,
    int instance_id
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- Processing instance " << instance_id << " ---\n";

    try {
        // 1. Resize mask if needed
        cv::Mat resized_mask = mask;
        if (mask.rows != depth.rows || mask.cols != depth.cols) {
            cv::resize(mask, resized_mask, depth.size(), 0, 0, cv::INTER_NEAREST);
        }

        // 2. GPU depth preprocessing (scale + mask)
        cv::Mat scaled_depth;
        if (config_.use_gpu && GPUDepth::isCudaAvailable()) {
            scaled_depth = GPUDepth::preprocess(depth, resized_mask, config_.depth.scale_to_meters);
        } else {
            // CPU fallback
            cv::Mat float_depth;
            depth.convertTo(float_depth, CV_32FC1, 1.0 / config_.depth.scale_to_meters);

            // Apply mask
            if (config_.segmentation.apply_mask && !resized_mask.empty()) {
                cv::Mat mask_bool;
                cv::threshold(resized_mask, mask_bool, 10, 255, cv::THRESH_BINARY);
                float_depth.setTo(0, mask_bool == 0);
            }
            scaled_depth = float_depth;
        }

        // Check for valid depth data
        if (cv::countNonZero(scaled_depth) == 0) {
            std::cerr << "Instance " << instance_id << ": empty depth after masking\n";
            return std::nullopt;
        }

        // 3. GPU point cloud generation
        float fx = intrinsics(0, 0), fy = intrinsics(1, 1);
        float cx = intrinsics(0, 2), cy = intrinsics(1, 2);

        PointCloud pcd;
        if (config_.use_gpu && GPUDepth::isCudaAvailable()) {
            pcd = GPUPointCloud::generate(scaled_depth, rgb, fx, fy, cx, cy);
        } else {
            // CPU deprojection
            for (int v = 0; v < scaled_depth.rows; ++v) {
                for (int u = 0; u < scaled_depth.cols; ++u) {
                    float z = scaled_depth.at<float>(v, u);
                    if (z <= 0 || z > config_.depth.clipping_max) continue;

                    float x = (u - cx) * z / fx;
                    float y = (v - cy) * z / fy;
                    pcd.points.push_back(Eigen::Vector3f(x, y, z));

                    if (!rgb.empty()) {
                        cv::Vec3b bgr = rgb.at<cv::Vec3b>(v, u);
                        pcd.colors.push_back(Eigen::Vector3f(
                            bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f));
                    }
                }
            }
        }

        if (pcd.empty()) {
            std::cerr << "Instance " << instance_id << ": empty point cloud\n";
            return std::nullopt;
        }
        std::cout << "Instance " << instance_id << ": " << pcd.size() << " points\n";

        // 4. Downsample and compute features
        PointCloud source_down = Registration::voxelDownsample(pcd, config_.registration.voxel_size);
        Registration::estimateNormals(source_down, 30);
        FPFHFeatures source_features = Registration::computeFPFH(
            source_down, config_.registration.voxel_size * 5.0f);

        // 5. RANSAC coarse registration
        RegistrationResult coarse = Registration::ransacRegistration(
            source_down, ref_cloud, source_features, ref_features,
            config_.registration.voxel_size,
            config_.registration.ransac_max_iterations,
            config_.registration.ransac_confidence
        );

        // 6. ICP refinement
        float icp_threshold = config_.registration.voxel_size * config_.registration.icp_distance_factor;
        RegistrationResult refined;

        if (config_.use_gpu && GPURegistration::isCudaAvailable()) {
            refined = GPURegistration::icpRefine(
                source_down, ref_cloud,
                coarse.transformation, icp_threshold,
                config_.registration.icp_max_iterations
            );
        } else {
            refined = Registration::icpRefine(
                source_down, ref_cloud,
                coarse.transformation, icp_threshold,
                config_.registration.icp_max_iterations,
                config_.registration.use_point_to_plane
            );
        }

        if (refined.fitness < config_.registration.min_fitness) {
            std::cerr << "Instance " << instance_id
                      << ": low fitness " << refined.fitness << "\n";
        }

        // 7. Compute world pose
        Eigen::Matrix4f T_camera_object = refined.transformation.inverse();
        Eigen::Matrix4f T_world_object = config_.camera_extrinsics * T_camera_object;

        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << "Instance " << instance_id << " done in " << ms << " ms "
                  << "(fitness=" << refined.fitness << ")\n";

        return T_world_object;

    } catch (const std::exception& e) {
        std::cerr << "Instance " << instance_id << " error: " << e.what() << "\n";
        return std::nullopt;
    }
}

// ─── Duplicate waypoint filter ───

std::vector<Eigen::Matrix4f> Pipeline::filterDuplicates(
    const std::vector<Eigen::Matrix4f>& waypoints,
    float min_distance
) {
    std::vector<Eigen::Matrix4f> filtered;

    for (const auto& wp : waypoints) {
        Eigen::Vector3f pos = wp.block<3,1>(0,3);
        bool is_dup = false;

        for (size_t i = 0; i < filtered.size(); ++i) {
            float dist = (pos - filtered[i].block<3,1>(0,3)).norm();
            if (dist < min_distance) {
                is_dup = true;
                // Keep the one closer to origin
                float existing_dist = filtered[i].block<3,1>(0,3).norm();
                float current_dist  = pos.norm();
                if (current_dist < existing_dist) {
                    filtered[i] = wp;  // Replace
                }
                break;
            }
        }
        if (!is_dup) filtered.push_back(wp);
    }

    std::cout << "Filtered: " << waypoints.size() << " → " << filtered.size() << " waypoints\n";
    return filtered;
}

// ─── Main pipeline ───

void Pipeline::run() {
    auto pipeline_start = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== Starting Pipeline ===\n";

    cv::Mat rgb, depth;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

    // 1. Acquisition
    if (config_.use_camera) {
        std::cout << "\n[1/5] Camera capture (RealSense)...\n";
        RealSenseCamera camera(config_.camera.width, config_.camera.height);
        if (!camera.connect() || !camera.capture(rgb, depth)) {
            std::cerr << "Camera capture failed.\n";
            return;
        }
        K = camera.getIntrinsics();
        camera.disconnect();
    } else {
        std::cout << "\n[1/5] Using dummy data...\n";
        
        // Try loading from file
        if (!config_.dummy_rgb_path.empty() && !config_.dummy_depth_path.empty()) {
            rgb = cv::imread(config_.dummy_rgb_path, cv::IMREAD_COLOR);
            depth = cv::imread(config_.dummy_depth_path, cv::IMREAD_UNCHANGED);
            
            // Default intrinsics for 1280x720 generic camera
            K << 900, 0, 640,
                 0, 900, 360,
                 0, 0, 1;
        } 
        
        // Procedural generation if loading failed or paths empty
        if (rgb.empty() || depth.empty()) {
            std::cout << "Generating procedural test scene...\n";
            int w = config_.camera.width;
            int h = config_.camera.height;
            rgb = cv::Mat(h, w, CV_8UC3, cv::Scalar(50, 50, 50));
            depth = cv::Mat(h, w, CV_16UC1, cv::Scalar(0));

            // Intrinsic parameters
            float fx = 900, fy = 900, cx = w/2.0f, cy = h/2.0f;
            K << fx, 0, cx,
                 0, fy, cy,
                 0, 0, 1;

            // Generate a floor plane at Z=1.0m and a box at Z=0.8m
            float floor_z = 1.0f;
            float box_z = 0.8f;
            
            for (int v = 0; v < h; ++v) {
                for (int u = 0; u < w; ++u) {
                    // Floor
                    float z = floor_z;
                    
                    // Box in center
                    if (abs(u - cx) < 100 && abs(v - cy) < 100) {
                        z = box_z;
                        rgb.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 255); // Red box
                    } else {
                        // Checkerboard floor
                        if (((u / 50) + (v / 50)) % 2 == 0) 
                            rgb.at<cv::Vec3b>(v, u) = cv::Vec3b(200, 200, 200);
                    }

                    // Convert meters to generic depth units (assuming scale 1000)
                    unsigned short d_val = static_cast<unsigned short>(z * config_.depth.scale_to_meters);
                    depth.at<unsigned short>(v, u) = d_val;
                }
            }
        }
    }

    if (rgb.empty() || depth.empty()) {
        std::cerr << "Failed to acquire or generate data.\n";
        return;
    }

    // 2. Segmentation
    std::cout << "\n[2/5] Segmentation...\n";
    // If using dummy data and no masks provided, generate a dummy mask for the center box
    std::vector<cv::Mat> masks;
    if (!config_.use_camera && config_.segmentation.masks_input_dir.empty()) {
        std::cout << "Generating dummy mask for box...\n";
        cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8UC1);
        int cx = depth.cols / 2;
        int cy = depth.rows / 2;
        cv::rectangle(mask, cv::Point(cx-100, cy-100), cv::Point(cx+100, cy+100), cv::Scalar(255), -1);
        masks.push_back(mask);
    } else {
        masks = Segmentation::getMasks(
            rgb,
            config_.segmentation.sam_server_url,
            config_.segmentation.sam_query,
            config_.segmentation.masks_input_dir
        );
    }

    if (masks.empty()) {
        std::cerr << "No segmentation masks found.\n";
        return;
    }
    std::cout << "Found " << masks.size() << " masks\n";

    // 3. Load reference model
    std::cout << "\n[3/5] Loading reference model...\n";
    PointCloud ref_cloud;
    if (config_.reference_model_path.empty() && !config_.use_camera) {
        // Generate a dummy reference cloud (a flat square matching the box top)
        std::cout << "Generating dummy reference model...\n";
        for (float x = -0.1f; x <= 0.1f; x += 0.005f) {
            for (float y = -0.1f; y <= 0.1f; y += 0.005f) {
                ref_cloud.points.emplace_back(x, y, 0.0f);
                ref_cloud.normals.emplace_back(0, 0, 1);
            }
        }
    } else {
        ref_cloud = Registration::loadReferenceModel(config_.reference_model_path);
    }
    
    if (ref_cloud.empty()) {
         std::cout << "Warning: Empty reference model. Registration may fail.\n";
    }

    PointCloud ref_down = Registration::voxelDownsample(ref_cloud, config_.registration.voxel_size);
    Registration::estimateNormals(ref_down, 30);
    FPFHFeatures ref_features = Registration::computeFPFH(
        ref_down, config_.registration.voxel_size * 5.0f);

    // 4. Start OpenGL viewer (if enabled)
    if (config_.viz_backend == VizBackend::OPENGL) {
        viewer_ = std::make_unique<GLViewer>();
        viewer_->start();
    }

    // 5. Process all instances in parallel
    std::cout << "\n[4/5] Processing " << masks.size() << " instances (parallel)...\n";
    auto proc_start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<std::optional<Eigen::Matrix4f>>> futures;
    for (size_t i = 0; i < masks.size(); ++i) {
        futures.push_back(thread_pool_->enqueue(
            &Pipeline::processInstance, this,
            masks[i], depth, rgb, K, ref_down, ref_features, static_cast<int>(i)
        ));
    }

    // Collect results
    std::vector<Eigen::Matrix4f> raw_waypoints;
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        if (result.has_value()) {
            raw_waypoints.push_back(result.value());

            // Update viewer
            if (viewer_ && viewer_->isRunning()) {
                viewer_->setPose("pose_" + std::to_string(i), result.value());
            }
        }
    }

    auto proc_end = std::chrono::high_resolution_clock::now();
    float proc_ms = std::chrono::duration<float, std::milli>(proc_end - proc_start).count();
    std::cout << "\nAll instances processed in " << proc_ms << " ms\n";

    // Filter duplicates
    auto final_waypoints = filterDuplicates(raw_waypoints, 0.1f);

    // Update viewer with path
    if (viewer_ && viewer_->isRunning() && !final_waypoints.empty()) {
        std::vector<Eigen::Vector3f> path_positions;
        for (const auto& wp : final_waypoints) {
            path_positions.push_back(wp.block<3,1>(0,3));
        }
        viewer_->setPath(path_positions);
    }

    // 6. Robot execution
    if (config_.use_robot) {
        std::cout << "\n[5/5] Robot execution...\n";
        Robot robot(config_.robot.ip);
        if (robot.connect()) {
            for (size_t i = 0; i < final_waypoints.size(); ++i) {
                std::cout << "\nPicking object " << (i + 1) << "/" << final_waypoints.size() << "\n";
                robot.pick(final_waypoints[i], config_.robot.approach_offset_z);
            }
            robot.disconnect();
        }
    } else {
        std::cout << "\n[5/5] Robot execution skipped (use_robot=false)\n";
        std::cout << "Computed " << final_waypoints.size() << " pick poses.\n";
    }

    auto pipeline_end = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(pipeline_end - pipeline_start).count();
    std::cout << "\n=== Pipeline complete: " << total_ms << " ms ===\n";

    // Keep viewer open if running
    if (viewer_ && viewer_->isRunning()) {
        std::cout << "Viewer is running. Close the window to exit.\n";
        while (viewer_->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

}  // namespace industry_picking
