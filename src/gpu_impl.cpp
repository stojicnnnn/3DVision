#include "gpu_depth.hpp"
#include "gpu_registration.hpp"

#include "../cuda/depth_processing.cuh"
#include "../cuda/pointcloud.cuh"
#include "../cuda/icp.cuh"

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#endif

namespace industry_picking {


bool GPUDepth::isCudaAvailable() {
#ifdef CUDA_AVAILABLE
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
#else
    return false;
#endif
}

cv::Mat GPUDepth::preprocess(
    const cv::Mat& raw_depth,
    const cv::Mat& mask,
    float scale
) {
#ifdef CUDA_AVAILABLE
    int width = raw_depth.cols;
    int height = raw_depth.rows;
    size_t num_pixels = width * height;

    unsigned short* d_raw = nullptr;
    unsigned char*  d_mask = nullptr;
    float*          d_out = nullptr;

    cudaMalloc(&d_raw, num_pixels * sizeof(unsigned short));
    if (!mask.empty()) {
        cudaMalloc(&d_mask, num_pixels * sizeof(unsigned char));
    }
    cudaMalloc(&d_out, num_pixels * sizeof(float));

    cudaMemcpy(d_raw, raw_depth.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    if (d_mask) {
        cudaMemcpy(d_mask, mask.data, num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    cuda::launchDepthPreprocess(d_raw, d_mask, d_out, width, height, scale, d_mask != nullptr);

    cv::Mat out_depth(height, width, CV_32FC1);
    cudaMemcpy(out_depth.data, d_out, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_raw);
    if (d_mask) cudaFree(d_mask);
    cudaFree(d_out);

    return out_depth;
#else
    throw std::runtime_error("CUDA not available");
#endif
}


PointCloud GPUPointCloud::generate(
    const cv::Mat& depth,
    const cv::Mat& rgb,
    float fx, float fy, float cx, float cy
) {
#ifdef CUDA_AVAILABLE
    int width = depth.cols;
    int height = depth.rows;
    int num_pixels = width * height;

    float* d_depth = nullptr;
    unsigned char* d_rgb = nullptr;
    float* d_points = nullptr;
    int* d_count = nullptr;

    cudaMalloc(&d_depth, num_pixels * sizeof(float));
    cudaMalloc(&d_rgb, num_pixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_points, num_pixels * 6 * sizeof(float)); // Max possible size
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_depth, depth.data, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
    if (!rgb.empty()) {
        cudaMemcpy(d_rgb, rgb.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_rgb, 255, num_pixels * 3); // White fallback
    }
    cudaMemset(d_count, 0, sizeof(int));

    float max_depth = 10.0f;
    cuda::launchDeproject(
        d_depth, d_rgb, d_points, d_count,
        width, height, fx, fy, cx, cy, max_depth
    );

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    PointCloud pcd;
    if (h_count > 0) {
        std::vector<float> h_points(h_count * 6);
        cudaMemcpy(h_points.data(), d_points, h_count * 6 * sizeof(float), cudaMemcpyDeviceToHost);

        pcd.points.resize(h_count);
        pcd.colors.resize(h_count);
        for (int i = 0; i < h_count; ++i) {
            pcd.points[i] = Eigen::Vector3f(h_points[6*i+0], h_points[6*i+1], h_points[6*i+2]);
            pcd.colors[i] = Eigen::Vector3f(h_points[6*i+3], h_points[6*i+4], h_points[6*i+5]);
        }
    }

    cudaFree(d_depth);
    cudaFree(d_rgb);
    cudaFree(d_points);
    cudaFree(d_count);

    return pcd;
#else
    return {};
#endif
}


bool GPURegistration::isCudaAvailable() {
#ifdef CUDA_AVAILABLE
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
#else
    return false;
#endif
}

RegistrationResult GPURegistration::icpRefine(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4f& initial_transform,
    float distance_threshold,
    int max_iterations
) {
#ifdef CUDA_AVAILABLE
    int n_src = static_cast<int>(source.size());
    int n_tgt = static_cast<int>(target.size());

    std::vector<float> h_source(n_src * 3), h_target(n_tgt * 3), h_normals(n_tgt * 3);
    for (int i = 0; i < n_src; ++i) {
        h_source[i*3+0] = source.points[i].x();
        h_source[i*3+1] = source.points[i].y();
        h_source[i*3+2] = source.points[i].z();
    }
    for (int i = 0; i < n_tgt; ++i) {
        h_target[i*3+0] = target.points[i].x();
        h_target[i*3+1] = target.points[i].y();
        h_target[i*3+2] = target.points[i].z();
        if (target.hasNormals()) {
            h_normals[i*3+0] = target.normals[i].x();
            h_normals[i*3+1] = target.normals[i].y();
            h_normals[i*3+2] = target.normals[i].z();
        }
    }

    float *d_source, *d_target, *d_normals, *d_transform;
    int *d_corr;
    float *d_dist, *d_JtJ, *d_Jtr;

    cudaMalloc(&d_source,    n_src * 3 * sizeof(float));
    cudaMalloc(&d_target,    n_tgt * 3 * sizeof(float));
    cudaMalloc(&d_normals,   n_tgt * 3 * sizeof(float));
    cudaMalloc(&d_transform, 16 * sizeof(float));
    cudaMalloc(&d_corr,      n_src * sizeof(int));
    cudaMalloc(&d_dist,      n_src * sizeof(float));
    cudaMalloc(&d_JtJ,       36 * sizeof(float));
    cudaMalloc(&d_Jtr,        6 * sizeof(float));

    cudaMemcpy(d_source,  h_source.data(),  n_src * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target,  h_target.data(),  n_tgt * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals.data(), n_tgt * 3 * sizeof(float), cudaMemcpyHostToDevice);

    Eigen::Matrix4f T = initial_transform;
    RegistrationResult result;
    result.transformation = T;

    for (int iter = 0; iter < max_iterations; ++iter) {
        float h_T[16];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                h_T[r * 4 + c] = T(r, c);
        cudaMemcpy(d_transform, h_T, 16 * sizeof(float), cudaMemcpyHostToDevice);

        cuda::launchFindCorrespondences(
            d_source, d_target, d_corr, d_dist,
            n_src, n_tgt, d_transform, distance_threshold
        );

        cuda::launchBuildLinearSystem(
            d_source, d_target, d_normals,
            d_corr, d_dist, d_JtJ, d_Jtr,
            n_src, distance_threshold
        );

        float h_JtJ[36], h_Jtr[6];
        cudaMemcpy(h_JtJ, d_JtJ, 36 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jtr, d_Jtr,  6 * sizeof(float), cudaMemcpyDeviceToHost);

        Eigen::Map<Eigen::Matrix<float, 6, 6, Eigen::RowMajor>> ATA(h_JtJ);
        Eigen::Map<Eigen::Matrix<float, 6, 1>> ATb(h_Jtr);

        Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(-ATb);

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = (Eigen::AngleAxisf(x(0), Eigen::Vector3f::UnitX())
                               * Eigen::AngleAxisf(x(1), Eigen::Vector3f::UnitY())
                               * Eigen::AngleAxisf(x(2), Eigen::Vector3f::UnitZ())).matrix();
        delta.block<3,1>(0,3) = x.tail<3>();

        T = delta * T;

        std::vector<int>   h_corr(n_src);
        std::vector<float> h_dists(n_src);
        cudaMemcpy(h_corr.data(),  d_corr, n_src * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dists.data(), d_dist, n_src * sizeof(float), cudaMemcpyDeviceToHost);

        int n_inlier = 0;
        float total_err = 0;
        float thresh_sq = distance_threshold * distance_threshold;
        for (int i = 0; i < n_src; ++i) {
            if (h_corr[i] >= 0 && h_dists[i] < thresh_sq) {
                ++n_inlier;
                total_err += h_dists[i];
            }
        }

        float prev_rmse = result.rmse;
        result.fitness = static_cast<float>(n_inlier) / n_src;
        result.rmse = n_inlier > 0 ? std::sqrt(total_err / n_inlier) : 999.0f;
        result.transformation = T;

        if (iter > 0 && std::abs(prev_rmse - result.rmse) < 1e-6f) {
            std::cout << "GPU ICP converged at iteration " << iter << "\n";
            break;
        }
    }

    cudaFree(d_source);  cudaFree(d_target);  cudaFree(d_normals);
    cudaFree(d_transform); cudaFree(d_corr); cudaFree(d_dist);
    cudaFree(d_JtJ);    cudaFree(d_Jtr);

    std::cout << "GPU ICP result: fitness=" << result.fitness << ", RMSE=" << result.rmse << "\n";
    return result;
#else
    throw std::runtime_error("CUDA not available");
#endif
}

}
