#include "registration.hpp"

#include <opencv2/imgcodecs.hpp>
#include <Eigen/Eigenvalues>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>

namespace industry_picking {

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h = std::hash<int>()(k.x);
        h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

PointCloud Registration::voxelDownsample(const PointCloud& cloud, float voxel_size) {
    std::unordered_map<VoxelKey, std::vector<size_t>, VoxelKeyHash> grid;

    float inv = 1.0f / voxel_size;
    for (size_t i = 0; i < cloud.size(); ++i) {
        VoxelKey key{
            static_cast<int>(std::floor(cloud.points[i].x() * inv)),
            static_cast<int>(std::floor(cloud.points[i].y() * inv)),
            static_cast<int>(std::floor(cloud.points[i].z() * inv))
        };
        grid[key].push_back(i);
    }

    PointCloud result;
    result.points.reserve(grid.size());
    if (cloud.hasColors()) result.colors.reserve(grid.size());

    for (auto& [key, indices] : grid) {
        Eigen::Vector3f avg_pt = Eigen::Vector3f::Zero();
        Eigen::Vector3f avg_col = Eigen::Vector3f::Zero();
        for (size_t idx : indices) {
            avg_pt += cloud.points[idx];
            if (cloud.hasColors()) avg_col += cloud.colors[idx];
        }
        float n = static_cast<float>(indices.size());
        result.points.push_back(avg_pt / n);
        if (cloud.hasColors()) result.colors.push_back(avg_col / n);
    }

    std::cout << "Voxel downsample: " << cloud.size() << " → " << result.size() << " points\n";
    return result;
}


static std::vector<size_t> findKNN(
    const std::vector<Eigen::Vector3f>& points,
    const Eigen::Vector3f& query,
    int k
) {
    std::vector<std::pair<float, size_t>> dists;
    dists.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        float d = (points[i] - query).squaredNorm();
        dists.emplace_back(d, i);
    }
    std::partial_sort(dists.begin(), dists.begin() + std::min(k, (int)dists.size()), dists.end());

    std::vector<size_t> result;
    for (int i = 0; i < std::min(k, (int)dists.size()); ++i) {
        result.push_back(dists[i].second);
    }
    return result;
}

static std::vector<size_t> findRadiusNN(
    const std::vector<Eigen::Vector3f>& points,
    const Eigen::Vector3f& query,
    float radius,
    int max_nn
) {
    float r2 = radius * radius;
    std::vector<std::pair<float, size_t>> dists;
    for (size_t i = 0; i < points.size(); ++i) {
        float d2 = (points[i] - query).squaredNorm();
        if (d2 <= r2) dists.emplace_back(d2, i);
    }
    std::sort(dists.begin(), dists.end());

    std::vector<size_t> result;
    for (int i = 0; i < std::min(max_nn, (int)dists.size()); ++i) {
        result.push_back(dists[i].second);
    }
    return result;
}


void Registration::estimateNormals(PointCloud& cloud, int k) {
    cloud.normals.resize(cloud.size());

    for (size_t i = 0; i < cloud.size(); ++i) {
        auto neighbors = findKNN(cloud.points, cloud.points[i], k);

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (size_t idx : neighbors) centroid += cloud.points[idx];
        centroid /= static_cast<float>(neighbors.size());

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (size_t idx : neighbors) {
            Eigen::Vector3f diff = cloud.points[idx] - centroid;
            cov += diff * diff.transpose();
        }
        cov /= static_cast<float>(neighbors.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        cloud.normals[i] = solver.eigenvectors().col(0);

        if (cloud.normals[i].dot(-cloud.points[i]) < 0) {
            cloud.normals[i] = -cloud.normals[i];
        }
    }
    std::cout << "Estimated normals for " << cloud.size() << " points\n";
}


FPFHFeatures Registration::computeFPFH(const PointCloud& cloud, float radius) {
    FPFHFeatures features;
    features.descriptors.resize(cloud.size());

    auto computeSPFH = [&](size_t idx) -> std::array<float, 33> {
        std::array<float, 33> hist{};
        auto neighbors = findRadiusNN(cloud.points, cloud.points[idx], radius, 100);

        for (size_t ni : neighbors) {
            if (ni == idx) continue;

            Eigen::Vector3f diff = cloud.points[ni] - cloud.points[idx];
            float dist = diff.norm();
            if (dist < 1e-8f) continue;

            Eigen::Vector3f u = cloud.normals[idx];
            Eigen::Vector3f v = u.cross(diff / dist);
            Eigen::Vector3f w = u.cross(v);

            float alpha = v.dot(cloud.normals[ni]);
            float phi   = u.dot(diff / dist);
            float theta = std::atan2(w.dot(cloud.normals[ni]), u.dot(cloud.normals[ni]));

            int bin_a = std::clamp(static_cast<int>((alpha + 1.0f) * 5.5f), 0, 10);
            int bin_p = std::clamp(static_cast<int>((phi + 1.0f)   * 5.5f), 0, 10);
            int bin_t = std::clamp(static_cast<int>((theta / M_PI + 1.0f) * 5.5f), 0, 10);

            hist[bin_a]      += 1.0f;
            hist[11 + bin_p] += 1.0f;
            hist[22 + bin_t] += 1.0f;
        }

        float sum = 0;
        for (float v : hist) sum += v;
        if (sum > 0) for (float& v : hist) v /= sum;

        return hist;
    };

    std::vector<std::array<float, 33>> spfh(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        spfh[i] = computeSPFH(i);
    }
    for (size_t i = 0; i < cloud.size(); ++i) {
        auto neighbors = findRadiusNN(cloud.points, cloud.points[i], radius, 100);
        std::array<float, 33> fpfh{};

        for (int d = 0; d < 33; ++d) fpfh[d] = spfh[i][d];

        for (size_t ni : neighbors) {
            if (ni == i) continue;
            float dist = (cloud.points[ni] - cloud.points[i]).norm();
            if (dist < 1e-8f) continue;
            float weight = 1.0f / dist;
            for (int d = 0; d < 33; ++d) {
                fpfh[d] += weight * spfh[ni][d];
            }
        }

        float sum = 0;
        for (float v : fpfh) sum += v;
        if (sum > 0) for (float& v : fpfh) v /= sum;

        features.descriptors[i] = fpfh;
    }

    std::cout << "Computed FPFH features for " << cloud.size() << " points\n";
    return features;
}


RegistrationResult Registration::ransacRegistration(
    const PointCloud& source,
    const PointCloud& target,
    const FPFHFeatures& source_features,
    const FPFHFeatures& target_features,
    float voxel_size,
    int max_iterations,
    float confidence
) {
    float distance_threshold = voxel_size * 1.5f;
    std::cout << "RANSAC registration (threshold=" << distance_threshold << ", max_iter=" << max_iterations << ")\n";

    std::vector<size_t> correspondences(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        float best_dist = std::numeric_limits<float>::max();
        size_t best_idx = 0;
        for (size_t j = 0; j < target.size(); ++j) {
            float dist = 0;
            for (int d = 0; d < 33; ++d) {
                float diff = source_features.descriptors[i][d] - target_features.descriptors[j][d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        correspondences[i] = best_idx;
    }

    RegistrationResult best_result;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, source.size() - 1);

    for (int iter = 0; iter < max_iterations; ++iter) {
        size_t i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Eigen::Matrix3f src_pts, tgt_pts;
        src_pts.col(0) = source.points[i0]; src_pts.col(1) = source.points[i1]; src_pts.col(2) = source.points[i2];
        tgt_pts.col(0) = target.points[correspondences[i0]];
        tgt_pts.col(1) = target.points[correspondences[i1]];
        tgt_pts.col(2) = target.points[correspondences[i2]];

        Eigen::Vector3f src_centroid = src_pts.rowwise().mean();
        Eigen::Vector3f tgt_centroid = tgt_pts.rowwise().mean();

        Eigen::Matrix3f src_centered = src_pts.colwise() - src_centroid;
        Eigen::Matrix3f tgt_centered = tgt_pts.colwise() - tgt_centroid;

        Eigen::Matrix3f H = src_centered * tgt_centered.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

        if (R.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }

        Eigen::Vector3f t = tgt_centroid - R * src_centroid;

        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3,3>(0,0) = R;
        transform.block<3,1>(0,3) = t;

        int inliers = 0;
        float total_error = 0;
        for (size_t i = 0; i < source.size(); ++i) {
            Eigen::Vector3f transformed = R * source.points[i] + t;
            float err = (transformed - target.points[correspondences[i]]).norm();
            if (err < distance_threshold) {
                ++inliers;
                total_error += err * err;
            }
        }

        float fitness = static_cast<float>(inliers) / source.size();
        float rmse = inliers > 0 ? std::sqrt(total_error / inliers) : 999.0f;

        if (fitness > best_result.fitness) {
            best_result.transformation = transform;
            best_result.fitness = fitness;
            best_result.rmse = rmse;
        }

        if (fitness > confidence) break;
    }

    std::cout << "RANSAC result: fitness=" << best_result.fitness << ", RMSE=" << best_result.rmse << "\n";
    return best_result;
}

RegistrationResult Registration::icpRefine(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4f& initial_transform,
    float distance_threshold,
    int max_iterations,
    bool point_to_plane
) {
    std::cout << "ICP refinement (threshold=" << distance_threshold
              << ", max_iter=" << max_iterations
              << ", mode=" << (point_to_plane ? "point-to-plane" : "point-to-point") << ")\n";

    Eigen::Matrix4f T = initial_transform;
    RegistrationResult result;
    result.transformation = T;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::Matrix3f R = T.block<3,3>(0,0);
        Eigen::Vector3f t = T.block<3,1>(0,3);

        int n_corr = 0;
        float total_error = 0;

        Eigen::Matrix<float, 6, 6> ATA = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> ATb = Eigen::Matrix<float, 6, 1>::Zero();

        std::vector<Eigen::Vector3f> src_corr, tgt_corr;

        for (size_t i = 0; i < source.size(); ++i) {
            Eigen::Vector3f p = R * source.points[i] + t;
            float best_dist2 = std::numeric_limits<float>::max();
            size_t best_idx = 0;
            for (size_t j = 0; j < target.size(); ++j) {
                float d2 = (p - target.points[j]).squaredNorm();
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_idx = j;
                }
            }

            float d = std::sqrt(best_dist2);
            if (d > distance_threshold) continue;

            ++n_corr;
            total_error += best_dist2;

            if (point_to_plane && target.hasNormals()) {
                const Eigen::Vector3f& q = target.points[best_idx];
                const Eigen::Vector3f& n = target.normals[best_idx];
                Eigen::Vector3f cross = p.cross(n);

                Eigen::Matrix<float, 1, 6> J;
                J << cross.x(), cross.y(), cross.z(), n.x(), n.y(), n.z();

                float residual = (p - q).dot(n);

                ATA += J.transpose() * J;
                ATb += J.transpose() * residual;
            } else {
                src_corr.push_back(p);
                tgt_corr.push_back(target.points[best_idx]);
            }
        }

        if (n_corr < 3) break;

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();

        if (point_to_plane && target.hasNormals()) {
            Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(-ATb);

            float a = x(0), b = x(1), g = x(2);
            delta.block<3,3>(0,0) = (Eigen::AngleAxisf(a, Eigen::Vector3f::UnitX())
                                   * Eigen::AngleAxisf(b, Eigen::Vector3f::UnitY())
                                   * Eigen::AngleAxisf(g, Eigen::Vector3f::UnitZ())).matrix();
            delta.block<3,1>(0,3) = x.tail<3>();
        } else {
            Eigen::Vector3f src_mean = Eigen::Vector3f::Zero();
            Eigen::Vector3f tgt_mean = Eigen::Vector3f::Zero();
            for (size_t i = 0; i < src_corr.size(); ++i) {
                src_mean += src_corr[i];
                tgt_mean += tgt_corr[i];
            }
            src_mean /= static_cast<float>(src_corr.size());
            tgt_mean /= static_cast<float>(tgt_corr.size());

            Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
            for (size_t i = 0; i < src_corr.size(); ++i) {
                H += (src_corr[i] - src_mean) * (tgt_corr[i] - tgt_mean).transpose();
            }

            Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f dR = svd.matrixV() * svd.matrixU().transpose();
            if (dR.determinant() < 0) {
                Eigen::Matrix3f V = svd.matrixV();
                V.col(2) *= -1;
                dR = V * svd.matrixU().transpose();
            }
            delta.block<3,3>(0,0) = dR;
            delta.block<3,1>(0,3) = tgt_mean - dR * src_mean;
        }

        T = delta * T;

        float prev_rmse = result.rmse;
        result.rmse = std::sqrt(total_error / n_corr);
        result.fitness = static_cast<float>(n_corr) / source.size();
        result.transformation = T;

        if (iter > 0 && std::abs(prev_rmse - result.rmse) < 1e-6f) {
            std::cout << "ICP converged at iteration " << iter << "\n";
            break;
        }
    }

    std::cout << "ICP result: fitness=" << result.fitness << ", RMSE=" << result.rmse << "\n";
    return result;
}

PointCloud Registration::loadReferenceModel(const std::string& path) {
    PointCloud cloud;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open reference model: " << path << "\n";
        return cloud;
    }

    std::string line;
    int vertex_count = 0;
    bool has_color = false;
    bool in_header = true;

    while (std::getline(file, line) && in_header) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        }
        if (line.find("red") != std::string::npos || line.find("diffuse_red") != std::string::npos) {
            has_color = true;
        }
        if (line == "end_header") {
            in_header = false;
        }
    }

    cloud.points.reserve(vertex_count);
    if (has_color) cloud.colors.reserve(vertex_count);

    for (int i = 0; i < vertex_count; ++i) {
        float x, y, z;
        file >> x >> y >> z;
        cloud.points.push_back(Eigen::Vector3f(x, y, z));

        if (has_color) {
            float r, g, b;
            file >> r >> g >> b;
            if (r > 1.0f) { r /= 255.0f; g /= 255.0f; b /= 255.0f; }
            cloud.colors.push_back(Eigen::Vector3f(r, g, b));
        }
        std::getline(file, line);
    }

    std::cout << "Loaded reference model: " << cloud.size() << " points from " << path << "\n";
    return cloud;
}

}
