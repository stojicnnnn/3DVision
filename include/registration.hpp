#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace industry_picking {

/**
 * @brief A simple point cloud represented as vectors of points, normals, and colors.
 */
struct PointCloud {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;  // RGB in [0, 1]

    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }
    bool hasNormals() const { return normals.size() == points.size(); }
    bool hasColors() const { return colors.size() == points.size(); }
};

/**
 * @brief FPFH feature descriptor for a point cloud.
 */
struct FPFHFeatures {
    std::vector<std::array<float, 33>> descriptors;  // 33-bin histograms
    size_t size() const { return descriptors.size(); }
};

/**
 * @brief Result of a registration (alignment) operation.
 */
struct RegistrationResult {
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    float fitness = 0.0f;   // Fraction of inlier correspondences
    float rmse    = 0.0f;   // Root mean square error of inliers
};

/**
 * @brief Point cloud registration: voxel downsampling, normals, FPFH, RANSAC, ICP.
 */
class Registration {
public:
    /**
     * @brief Voxel grid downsampling.
     * @param cloud      Input point cloud
     * @param voxel_size Grid cell size in meters
     * @return Downsampled point cloud
     */
    static PointCloud voxelDownsample(const PointCloud& cloud, float voxel_size);

    /**
     * @brief Estimate surface normals using KNN + PCA.
     * @param cloud   Point cloud (modified in place)
     * @param k       Number of nearest neighbors
     */
    static void estimateNormals(PointCloud& cloud, int k = 30);

    /**
     * @brief Compute FPFH feature descriptors.
     * @param cloud   Point cloud with normals
     * @param radius  Search radius for FPFH computation
     * @return FPFH features (33-dim per point)
     */
    static FPFHFeatures computeFPFH(const PointCloud& cloud, float radius);

    /**
     * @brief RANSAC-based global registration using FPFH features.
     */
    static RegistrationResult ransacRegistration(
        const PointCloud& source,
        const PointCloud& target,
        const FPFHFeatures& source_features,
        const FPFHFeatures& target_features,
        float voxel_size,
        int max_iterations = 100000,
        float confidence   = 0.999f
    );

    /**
     * @brief ICP refinement (point-to-plane or point-to-point).
     */
    static RegistrationResult icpRefine(
        const PointCloud& source,
        const PointCloud& target,
        const Eigen::Matrix4f& initial_transform,
        float distance_threshold,
        int max_iterations     = 200,
        bool point_to_plane    = true
    );

    /**
     * @brief Load a reference point cloud model from a PLY/PCD file.
     */
    static PointCloud loadReferenceModel(const std::string& path);
};

}  // namespace industry_picking
