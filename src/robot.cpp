#include "robot.hpp"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>

namespace industry_picking {

Robot::Robot(const std::string& ip) : ip_(ip) {
    std::cout << "Robot created for IP: " << ip << "\n";
}

Robot::~Robot() {
    if (connected_) disconnect();
}

bool Robot::connect() {
    std::cout << "Connecting to xArm at " << ip_ << "...\n";

    connected_ = true;
    std::cout << "xArm connected (simulation mode).\n";
    return true;
}

void Robot::disconnect() {
    if (connected_) {
        connected_ = false;
        std::cout << "xArm disconnected.\n";
    }
}

bool Robot::move(const Eigen::Matrix4f& pose, int speed, bool wait) {
    if (!connected_) {
        std::cerr << "Robot not connected.\n";
        return false;
    }

    float x = pose(0, 3) * 1000.0f;
    float y = pose(1, 3) * 1000.0f;
    float z = pose(2, 3) * 1000.0f;

    Eigen::Matrix3f R = pose.block<3,3>(0,0);

    float pitch = std::asin(-R(2, 0));
    float roll, yaw;
    if (std::abs(R(2, 0)) < 0.999f) {
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw  = std::atan2(R(1, 0), R(0, 0));
    } else {
        roll = std::atan2(-R(1, 2), R(1, 1));
        yaw  = 0.0f;
    }

    float roll_deg  = roll  * 180.0f / M_PI;
    float pitch_deg = pitch * 180.0f / M_PI;
    float yaw_deg   = yaw   * 180.0f / M_PI;

    std::cout << "Moving to: [" << x << ", " << y << ", " << z
              << "] mm, RPY=[" << roll_deg << ", " << pitch_deg << ", " << yaw_deg << "] deg"
              << " speed=" << speed << "\n";

    if (wait) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return true;
}

Eigen::Matrix4f Robot::getPose() const {
    return Eigen::Matrix4f::Identity();
}

void Robot::closeGripper() {
    std::cout << "Closing gripper.\n";
}

void Robot::openGripper() {
    std::cout << "Opening gripper.\n";
}

bool Robot::pick(const Eigen::Matrix4f& pose, float approach_offset_z) {
    if (!connected_) return false;

    Eigen::Matrix4f offset = Eigen::Matrix4f::Identity();
    offset(2, 3) = approach_offset_z;
    Eigen::Matrix4f approach_pose = pose * offset;

    std::cout << "Moving to approach position...\n";
    move(approach_pose);

    Eigen::Matrix4f pick_offset = Eigen::Matrix4f::Identity();
    pick_offset(2, 3) = -0.001f;
    Eigen::Matrix4f pick_pose = pose * pick_offset;

    std::cout << "Descending to pick position...\n";
    move(pick_pose, 10);  // Slow approach

    closeGripper();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Retracting...\n";
    move(approach_pose);

    std::cout << "Pick completed.\n";
    return true;
}

}
