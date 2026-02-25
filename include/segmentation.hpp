#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <optional>

namespace industry_picking {

class Segmentation {
public:
    static std::vector<cv::Mat> loadMasksFromDir(const std::string& masks_dir);

    static std::vector<cv::Mat> getMasksFromSAM(
        const cv::Mat& rgb_image,
        const std::string& server_url,
        const std::string& query
    );

    static std::vector<cv::Mat> getMasks(
        const cv::Mat& rgb_image,
        const std::string& sam_server_url,
        const std::string& sam_query,
        const std::string& masks_dir
    );
};

}
