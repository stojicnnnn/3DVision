#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace industry_picking {

struct PointCloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;

    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }
    bool hasNormals() const { return normals.size() == points.size(); }
    bool hasColors() const { return colors.size() == points.size(); }
};

struct FPFHFeatures {
    std::vector<std::array<float, 33>> descriptors;
    size_t size() const { return descriptors.size(); }
};

struct RegistrationResult {
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    float fitness = 0.0f;
    float rmse    = 0.0f;
};

class Registration {
public:
    static PointCloud voxelDownsample(const PointCloud& cloud, float voxel_size);

    static void estimateNormals(PointCloud& cloud, int k = 30);

    static FPFHFeatures computeFPFH(const PointCloud& cloud, float radius);

    static RegistrationResult ransacRegistration(
        const PointCloud& source,
        const PointCloud& target,
        const FPFHFeatures& source_features,
        const FPFHFeatures& target_features,
        float voxel_size,
        int max_iterations = 100000,
        float confidence   = 0.999f
    );

    static RegistrationResult icpRefine(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initial_transform,
        float distance_threshold,
        int max_iterations     = 200,
        bool point_to_plane    = true
    );

    static PointCloud loadReferenceModel(const std::string& path);
};

}
