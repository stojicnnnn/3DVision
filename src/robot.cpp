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
    // TODO: Replace with actual xArm C++ SDK connection
    // For now, simulate connection
    std::cout << "Connecting to xArm at " << ip_ << "...\n";

    // In real implementation:
    // arm_handle_ = new XArmAPI(ip_);
    // ((XArmAPI*)arm_handle_)->connect();
    // ((XArmAPI*)arm_handle_)->motion_enable(true);
    // ((XArmAPI*)arm_handle_)->set_mode(0);
    // ((XArmAPI*)arm_handle_)->set_state(0);

    connected_ = true;
    std::cout << "xArm connected (simulation mode).\n";
    return true;
}

void Robot::disconnect() {
    if (connected_) {
        // TODO: actual disconnect
        connected_ = false;
        std::cout << "xArm disconnected.\n";
    }
}

bool Robot::move(const Eigen::Matrix4f& pose, int speed, bool wait) {
    if (!connected_) {
        std::cerr << "Robot not connected.\n";
        return false;
    }

    // Extract translation (meters → mm for xArm)
    float x = pose(0, 3) * 1000.0f;
    float y = pose(1, 3) * 1000.0f;
    float z = pose(2, 3) * 1000.0f;

    // Extract Euler angles from rotation matrix
    Eigen::Matrix3f R = pose.block<3,3>(0,0);

    // Roll, pitch, yaw from rotation matrix (XYZ intrinsic convention)
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

    // TODO: Replace with actual xArm SDK call:
    // ((XArmAPI*)arm_handle_)->set_position(x, y, z, roll_deg, pitch_deg, yaw_deg, speed, wait);

    if (wait) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return true;
}

Eigen::Matrix4f Robot::getPose() const {
    // TODO: actual implementation
    return Eigen::Matrix4f::Identity();
}

void Robot::closeGripper() {
    std::cout << "Closing gripper.\n";
    // TODO: ((XArmAPI*)arm_handle_)->close_lite6_gripper();
}

void Robot::openGripper() {
    std::cout << "Opening gripper.\n";
    // TODO: ((XArmAPI*)arm_handle_)->open_lite6_gripper();
}

bool Robot::pick(const Eigen::Matrix4f& pose, float approach_offset_z) {
    if (!connected_) return false;

    // 1. Move to approach position (above the object)
    Eigen::Matrix4f offset = Eigen::Matrix4f::Identity();
    offset(2, 3) = approach_offset_z;
    Eigen::Matrix4f approach_pose = pose * offset;

    std::cout << "Moving to approach position...\n";
    move(approach_pose);

    // 2. Descend to pick position
    Eigen::Matrix4f pick_offset = Eigen::Matrix4f::Identity();
    pick_offset(2, 3) = -0.001f;
    Eigen::Matrix4f pick_pose = pose * pick_offset;

    std::cout << "Descending to pick position...\n";
    move(pick_pose, 10);  // Slow approach

    // 3. Close gripper
    closeGripper();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 4. Retract
    std::cout << "Retracting...\n";
    move(approach_pose);

    std::cout << "Pick completed.\n";
    return true;
}

}  // namespace industry_picking
