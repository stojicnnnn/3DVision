#include "segmentation.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace industry_picking {

std::vector<cv::Mat> Segmentation::loadMasksFromDir(const std::string& masks_dir) {
    std::vector<cv::Mat> masks;

    if (!fs::is_directory(masks_dir)) {
        std::cerr << "Mask directory not found: " << masks_dir << "\n";
        return masks;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(masks_dir)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        cv::Mat mask = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);
        if (!mask.empty()) {
            // Threshold to binary
            cv::Mat binary;
            cv::threshold(mask, binary, 10, 255, cv::THRESH_BINARY);
            masks.push_back(binary);
        }
    }

    std::cout << "Loaded " << masks.size() << " masks from " << masks_dir << "\n";
    return masks;
}

std::vector<cv::Mat> Segmentation::getMasksFromSAM(
    const cv::Mat& rgb_image,
    const std::string& server_url,
    const std::string& query
) {
    std::cerr << "SAM server integration not yet implemented in C++.\n";
    std::cerr << "Falling back to local mask loading.\n";
    return {};
}

std::vector<cv::Mat> Segmentation::getMasks(
    const cv::Mat& rgb_image,
    const std::string& sam_server_url,
    const std::string& sam_query,
    const std::string& masks_dir
) {
    if (!sam_server_url.empty()) {
        auto masks = getMasksFromSAM(rgb_image, sam_server_url, sam_query);
        if (!masks.empty()) return masks;
    }

    return loadMasksFromDir(masks_dir);
}

}
