#pragma once

#include <Eigen/Core>
#include <string>

namespace industry_picking {

class Robot {
public:
    explicit Robot(const std::string& ip);
    ~Robot();

    Robot(const Robot&) = delete;
    Robot& operator=(const Robot&) = delete;

    bool connect();
    void disconnect();
    bool move(const Eigen::Matrix4f& pose, int speed = 80, bool wait = true);
    Eigen::Matrix4f getPose() const;
    void closeGripper();
    void openGripper();
    bool pick(const Eigen::Matrix4f& pose, float approach_offset_z = -0.101f);
    bool isConnected() const { return connected_; }

private:
    std::string ip_;
    bool connected_ = false;
    void* arm_handle_ = nullptr;
};

}
