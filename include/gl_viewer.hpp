#pragma once

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <Eigen/Core>
#include "registration.hpp"

namespace industry_picking {

class GLViewer {
public:
    GLViewer(int width = 1280, int height = 720, const std::string& title = "Industry Picking");
    ~GLViewer();

    GLViewer(const GLViewer&) = delete;
    GLViewer& operator=(const GLViewer&) = delete;

    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }

    void setPointCloud(const std::string& name, const PointCloud& cloud);
    void setPose(const std::string& name, const Eigen::Matrix4f& pose, float length = 0.05f);
    void setPath(const std::vector<Eigen::Vector3f>& waypoints);
    void clear();

private:
    void renderLoop();
    void initGL();
    void cleanupGL();
    void processInput();
    void render();

    unsigned int compileShader(const std::string& vertPath, const std::string& fragPath);
    unsigned int loadShaderFromFile(const std::string& path, unsigned int type);

    int width_, height_;
    std::string title_;
    void* window_ = nullptr;

    float cam_distance_  = 1.5f;
    float cam_yaw_       = 0.0f;
    float cam_pitch_     = 30.0f;
    Eigen::Vector3f cam_target_ = Eigen::Vector3f(0.0f, 0.0f, 0.9f);
    bool mouse_dragging_ = false;
    double last_mouse_x_ = 0, last_mouse_y_ = 0;

    struct RenderableCloud {
        std::vector<float> vertices;
        unsigned int vao = 0, vbo = 0;
        size_t count = 0;
        bool dirty = true;
    };

    struct RenderablePose {
        Eigen::Matrix4f transform;
        float length;
        bool dirty = true;
    };

    std::mutex data_mutex_;
    std::map<std::string, RenderableCloud> clouds_;
    std::map<std::string, RenderablePose>  poses_;
    std::vector<Eigen::Vector3f> path_;
    bool path_dirty_ = false;

    unsigned int cloud_shader_ = 0;
    unsigned int line_shader_  = 0;

    unsigned int path_vao_ = 0, path_vbo_ = 0;
    size_t path_count_ = 0;

    unsigned int axes_vao_ = 0, axes_vbo_ = 0;

    std::thread render_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
};

}
