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
    throw std::runtime_error("GPU ICP: use CPU fallback for now");
#else
    throw std::runtime_error("CUDA not available");
#endif
}

}
