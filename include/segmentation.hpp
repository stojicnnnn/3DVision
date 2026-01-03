#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <optional>

namespace industry_picking {

/**
 * @brief Loads segmentation masks from a local directory or a remote SAM server.
 */
class Segmentation {
public:
    /**
     * @brief Load binary masks from a directory of PNG images.
     * @param masks_dir  Path to directory containing mask images
     * @return Vector of binary masks (CV_8UC1, 255 = foreground)
     */
    static std::vector<cv::Mat> loadMasksFromDir(const std::string& masks_dir);

    /**
     * @brief Request masks from a remote SAM server via HTTP POST.
     * @param rgb_image    The RGB image to segment
     * @param server_url   URL of the SAM server
     * @param query        Text prompt for SAM
     * @return Vector of binary masks, or empty on failure
     */
    static std::vector<cv::Mat> getMasksFromSAM(
        const cv::Mat& rgb_image,
        const std::string& server_url,
        const std::string& query
    );

    /**
     * @brief Get masks — tries SAM server first, falls back to local directory.
     */
    static std::vector<cv::Mat> getMasks(
        const cv::Mat& rgb_image,
        const std::string& sam_server_url,
        const std::string& sam_query,
        const std::string& masks_dir
    );
};

}  // namespace industry_picking
