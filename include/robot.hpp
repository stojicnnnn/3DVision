#pragma once

#include <Eigen/Core>
#include <string>

namespace industry_picking {

/**
 * @brief Wrapper for UFactory xArm robot control.
 *
 * Communicates with the xArm via its C++ SDK or raw TCP protocol.
 * All poses are 4x4 homogeneous transformation matrices in meters.
 */
class Robot {
public:
    explicit Robot(const std::string& ip);
    ~Robot();

    // Non-copyable
    Robot(const Robot&) = delete;
    Robot& operator=(const Robot&) = delete;

    /** @brief Connect to the robot and enable motion. */
    bool connect();

    /** @brief Disconnect from the robot. */
    void disconnect();

    /**
     * @brief Move the robot to a pose.
     * @param pose   4x4 transformation matrix (meters, radians)
     * @param speed  Motion speed in mm/s (default 80)
     * @param wait   Block until motion completes
     */
    bool move(const Eigen::Matrix4f& pose, int speed = 80, bool wait = true);

    /** @brief Get the current robot pose as a 4x4 matrix. */
    Eigen::Matrix4f getPose() const;

    /** @brief Close the gripper. */
    void closeGripper();

    /** @brief Open the gripper. */
    void openGripper();

    /**
     * @brief Execute a pick sequence: approach → descend → grip → retract.
     * @param pose               Target object pose
     * @param approach_offset_z  Z-offset for approach position (meters, negative = above)
     */
    bool pick(const Eigen::Matrix4f& pose, float approach_offset_z = -0.101f);

    bool isConnected() const { return connected_; }

private:
    std::string ip_;
    bool connected_ = false;

    // Placeholder for actual xArm API handle
    // In a real implementation, this would be an XArmAPI pointer or socket
    void* arm_handle_ = nullptr;
};

}  // namespace industry_picking
