#include "gl_viewer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace industry_picking {


static GLViewer* g_viewer = nullptr;

static void scrollCallback(GLFWwindow*, double, double yoffset) {
    if (!g_viewer) return;
}


GLViewer::GLViewer(int width, int height, const std::string& title)
    : width_(width), height_(height), title_(title) {
}

GLViewer::~GLViewer() {
    stop();
}


bool GLViewer::start() {
    if (running_.load()) return true;

    should_stop_.store(false);
    render_thread_ = std::thread(&GLViewer::renderLoop, this);
    return true;
}

void GLViewer::stop() {
    should_stop_.store(true);
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
    running_.store(false);
}


void GLViewer::setPointCloud(const std::string& name, const PointCloud& cloud) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto& rc = clouds_[name];
    rc.vertices.clear();
    rc.vertices.reserve(cloud.size() * 6);

    for (size_t i = 0; i < cloud.size(); ++i) {
        rc.vertices.push_back(cloud.points[i].x());
        rc.vertices.push_back(cloud.points[i].y());
        rc.vertices.push_back(cloud.points[i].z());
        if (cloud.hasColors()) {
            rc.vertices.push_back(cloud.colors[i].x());
            rc.vertices.push_back(cloud.colors[i].y());
            rc.vertices.push_back(cloud.colors[i].z());
        } else {
            rc.vertices.push_back(0.8f);
            rc.vertices.push_back(0.8f);
            rc.vertices.push_back(0.8f);
        }
    }
    rc.count = cloud.size();
    rc.dirty = true;
}

void GLViewer::setPose(const std::string& name, const Eigen::Matrix4f& pose, float length) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    poses_[name] = {pose, length, true};
}

void GLViewer::setPath(const std::vector<Eigen::Vector3f>& waypoints) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    path_ = waypoints;
    path_dirty_ = true;
}

void GLViewer::clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    clouds_.clear();
    poses_.clear();
    path_.clear();
}


unsigned int GLViewer::loadShaderFromFile(const std::string& path, unsigned int type) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open shader: " << path << "\n";
        return 0;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string code = ss.str();
    const char* src = code.c_str();

    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compilation error (" << path << "): " << log << "\n";
        return 0;
    }
    return shader;
}

unsigned int GLViewer::compileShader(const std::string& vertPath, const std::string& fragPath) {
    unsigned int vert = loadShaderFromFile(vertPath, GL_VERTEX_SHADER);
    unsigned int frag = loadShaderFromFile(fragPath, GL_FRAGMENT_SHADER);
    if (!vert || !frag) return 0;

    unsigned int program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << "\n";
        return 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}


void GLViewer::renderLoop() {
    g_viewer = this;
    running_.store(true);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        running_.store(false);
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto* win = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!win) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        running_.store(false);
        return;
    }
    window_ = win;
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        glfwDestroyWindow(win);
        glfwTerminate();
        running_.store(false);
        return;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    cloud_shader_ = compileShader("shaders/pointcloud.vert", "shaders/pointcloud.frag");
    line_shader_  = compileShader("shaders/axes.vert", "shaders/axes.frag");

    glfwSetScrollCallback(win, [](GLFWwindow*, double, double yoffset) {
        if (g_viewer) g_viewer->cam_distance_ -= static_cast<float>(yoffset) * 0.1f;
        if (g_viewer) g_viewer->cam_distance_ = std::max(0.1f, g_viewer->cam_distance_);
    });

    while (!glfwWindowShouldClose(win) && !should_stop_.load()) {
        glfwPollEvents();
        processInput();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render();

        glfwSwapBuffers(win);
    }

    cleanupGL();
    glfwDestroyWindow(win);
    glfwTerminate();
    window_ = nullptr;
    running_.store(false);
    g_viewer = nullptr;
}

void GLViewer::processInput() {
    auto* win = static_cast<GLFWwindow*>(window_);
    double mx, my;
    glfwGetCursorPos(win, &mx, &my);

    bool left = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool middle = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    if (left) {
        if (mouse_dragging_) {
            float dx = static_cast<float>(mx - last_mouse_x_);
            float dy = static_cast<float>(my - last_mouse_y_);
            cam_yaw_   += dx * 0.3f;
            cam_pitch_ += dy * 0.3f;
            cam_pitch_ = std::clamp(cam_pitch_, -89.0f, 89.0f);
        }
        mouse_dragging_ = true;
    } else if (middle) {
        if (mouse_dragging_) {
            float dx = static_cast<float>(mx - last_mouse_x_) * 0.001f * cam_distance_;
            float dy = static_cast<float>(my - last_mouse_y_) * 0.001f * cam_distance_;
            float yaw_rad = cam_yaw_ * M_PI / 180.0f;
            cam_target_.x() -= dx * std::cos(yaw_rad);
            cam_target_.z() -= dx * std::sin(yaw_rad);
            cam_target_.y() += dy;
        }
        mouse_dragging_ = true;
    } else {
        mouse_dragging_ = false;
    }
    last_mouse_x_ = mx;
    last_mouse_y_ = my;
}

void GLViewer::render() {
    auto* win = static_cast<GLFWwindow*>(window_);
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    if (w == 0 || h == 0) return;
    glViewport(0, 0, w, h);

    float yaw_rad   = cam_yaw_   * M_PI / 180.0f;
    float pitch_rad = cam_pitch_ * M_PI / 180.0f;

    glm::vec3 eye(
        cam_target_.x() + cam_distance_ * cos(pitch_rad) * sin(yaw_rad),
        cam_target_.y() + cam_distance_ * sin(pitch_rad),
        cam_target_.z() + cam_distance_ * cos(pitch_rad) * cos(yaw_rad)
    );
    glm::vec3 target(cam_target_.x(), cam_target_.y(), cam_target_.z());
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)w / h, 0.01f, 100.0f);
    glm::mat4 vp = proj * view;

    std::lock_guard<std::mutex> lock(data_mutex_);

    if (cloud_shader_) {
        glUseProgram(cloud_shader_);
        int vpLoc = glGetUniformLocation(cloud_shader_, "uVP");
        glUniformMatrix4fv(vpLoc, 1, GL_FALSE, glm::value_ptr(vp));

        for (auto& [name, rc] : clouds_) {
            if (rc.count == 0) continue;

            if (rc.dirty) {
                if (rc.vao == 0) {
                    glGenVertexArrays(1, &rc.vao);
                    glGenBuffers(1, &rc.vbo);
                }
                glBindVertexArray(rc.vao);
                glBindBuffer(GL_ARRAY_BUFFER, rc.vbo);
                glBufferData(GL_ARRAY_BUFFER,
                    rc.vertices.size() * sizeof(float),
                    rc.vertices.data(), GL_DYNAMIC_DRAW);
                    
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(0);

                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
                glEnableVertexAttribArray(1);

                rc.dirty = false;
            }

            glBindVertexArray(rc.vao);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(rc.count));
        }
    }

    if (line_shader_) {
        glUseProgram(line_shader_);
        int vpLoc = glGetUniformLocation(line_shader_, "uVP");
        glUniformMatrix4fv(vpLoc, 1, GL_FALSE, glm::value_ptr(vp));

        for (auto& [name, rp] : poses_) {
            Eigen::Vector3f origin = rp.transform.block<3,1>(0,3);
            Eigen::Vector3f x_axis = origin + rp.transform.block<3,1>(0,0) * rp.length;
            Eigen::Vector3f y_axis = origin + rp.transform.block<3,1>(0,1) * rp.length;
            Eigen::Vector3f z_axis = origin + rp.transform.block<3,1>(0,2) * rp.length;

            float lines[] = {
                origin.x(), origin.y(), origin.z(), 1.0f, 0.0f, 0.0f,
                x_axis.x(), x_axis.y(), x_axis.z(), 1.0f, 0.0f, 0.0f,
                origin.x(), origin.y(), origin.z(), 0.0f, 1.0f, 0.0f,
                y_axis.x(), y_axis.y(), y_axis.z(), 0.0f, 1.0f, 0.0f,
                origin.x(), origin.y(), origin.z(), 0.0f, 0.0f, 1.0f,
                z_axis.x(), z_axis.y(), z_axis.z(), 0.0f, 0.0f, 1.0f,
            };

            if (axes_vao_ == 0) {
                glGenVertexArrays(1, &axes_vao_);
                glGenBuffers(1, &axes_vbo_);
            }
            glBindVertexArray(axes_vao_);
            glBindBuffer(GL_ARRAY_BUFFER, axes_vbo_);
            glBufferData(GL_ARRAY_BUFFER, sizeof(lines), lines, GL_DYNAMIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, 6);
        }

        if (!path_.empty()) {
            std::vector<float> path_verts;
            path_verts.reserve(path_.size() * 6);
            for (const auto& p : path_) {
                path_verts.push_back(p.x());
                path_verts.push_back(p.y());
                path_verts.push_back(p.z());
                path_verts.push_back(1.0f);
                path_verts.push_back(1.0f);
                path_verts.push_back(0.0f);
            }

            if (path_vao_ == 0) {
                glGenVertexArrays(1, &path_vao_);
                glGenBuffers(1, &path_vbo_);
            }
            glBindVertexArray(path_vao_);
            glBindBuffer(GL_ARRAY_BUFFER, path_vbo_);
            glBufferData(GL_ARRAY_BUFFER,
                path_verts.size() * sizeof(float),
                path_verts.data(), GL_DYNAMIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glLineWidth(3.0f);
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(path_.size()));
        }
    }

    glBindVertexArray(0);
}

void GLViewer::initGL() {
    // Already done in renderLoop
}

void GLViewer::cleanupGL() {
    for (auto& [name, rc] : clouds_) {
        if (rc.vao) glDeleteVertexArrays(1, &rc.vao);
        if (rc.vbo) glDeleteBuffers(1, &rc.vbo);
    }
    if (axes_vao_) glDeleteVertexArrays(1, &axes_vao_);
    if (axes_vbo_) glDeleteBuffers(1, &axes_vbo_);
    if (path_vao_) glDeleteVertexArrays(1, &path_vao_);
    if (path_vbo_) glDeleteBuffers(1, &path_vbo_);

    if (cloud_shader_) glDeleteProgram(cloud_shader_);
    if (line_shader_) glDeleteProgram(line_shader_);
}

}
