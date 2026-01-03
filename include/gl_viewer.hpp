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

/**
 * @brief Real-time OpenGL 3D viewer using GLFW + GLEW.
 *
 * Runs rendering in a background thread. Thread-safe methods to add/update
 * point clouds, pose frames, and paths. Supports orbit camera controls.
 */
class GLViewer {
public:
    GLViewer(int width = 1280, int height = 720, const std::string& title = "Industry Picking");
    ~GLViewer();

    // Non-copyable
    GLViewer(const GLViewer&) = delete;
    GLViewer& operator=(const GLViewer&) = delete;

    /** @brief Start the viewer in a background thread. */
    bool start();

    /** @brief Stop the viewer and close the window. */
    void stop();

    /** @brief Check if the viewer window is still open. */
    bool isRunning() const { return running_.load(); }

    // ─── Thread-safe data update methods ───

    /**
     * @brief Add or update a named point cloud.
     * @param name    Unique identifier
     * @param cloud   Point cloud data (copied)
     */
    void setPointCloud(const std::string& name, const PointCloud& cloud);

    /**
     * @brief Add or update a named coordinate frame (pose).
     * @param name    Unique identifier
     * @param pose    4x4 transformation matrix
     * @param length  Axis length in meters
     */
    void setPose(const std::string& name, const Eigen::Matrix4f& pose, float length = 0.05f);

    /**
     * @brief Set the waypoint path (line strip).
     * @param waypoints  Ordered list of 3D positions
     */
    void setPath(const std::vector<Eigen::Vector3f>& waypoints);

    /** @brief Clear all rendered objects. */
    void clear();

private:
    void renderLoop();
    void initGL();
    void cleanupGL();
    void processInput();
    void render();

    // ─── Shader compilation ───
    unsigned int compileShader(const std::string& vertPath, const std::string& fragPath);
    unsigned int loadShaderFromFile(const std::string& path, unsigned int type);

    // ─── Window ───
    int width_, height_;
    std::string title_;
    void* window_ = nullptr;  // GLFWwindow*

    // ─── Camera (orbit) ───
    float cam_distance_  = 1.0f;
    float cam_yaw_       = 0.0f;
    float cam_pitch_     = 30.0f;
    Eigen::Vector3f cam_target_ = Eigen::Vector3f::Zero();
    bool mouse_dragging_ = false;
    double last_mouse_x_ = 0, last_mouse_y_ = 0;

    // ─── Rendering data (protected by mutex) ───
    struct RenderableCloud {
        std::vector<float> vertices;  // Interleaved: x,y,z,r,g,b
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

    // Shader programs
    unsigned int cloud_shader_ = 0;
    unsigned int line_shader_  = 0;

    // Path VAO/VBO
    unsigned int path_vao_ = 0, path_vbo_ = 0;
    size_t path_count_ = 0;

    // Axes VAO/VBO
    unsigned int axes_vao_ = 0, axes_vbo_ = 0;

    // ─── Threading ───
    std::thread render_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
};

}  // namespace industry_picking
