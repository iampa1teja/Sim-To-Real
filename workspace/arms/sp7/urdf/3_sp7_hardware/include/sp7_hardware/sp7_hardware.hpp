#ifndef SP7_HARDWARE_INTERFACE_HPP
#define SP7_HARDWARE_INTERFACE_HPP

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "st_sc_servo_control_lib/SCServo.h"
#include <cmath>

#define SLEEP_TIME_AFTER_BEGIN 100000
#define SLEEP_TIME_ACTIVATE_DEACTIVATE 4096000
#define SLEEP_TIME_TOGGLE_TORQUE 50000

#define RAW_POS_TO_RAD(raw_pos) ((raw_pos / 4095.0 - 0.5) * 2.0 * M_PI)
#define RAW_SPD_TO_RAD_PER_SEC(raw_spd) (raw_spd * 0.001533980787885641)

namespace sp7_hardware
{

    class SP7HardwareInterface : public hardware_interface::SystemInterface
    {
    public:
        hardware_interface::CallbackReturn
        on_init(const hardware_interface::HardwareComponentInterfaceParams &params) override;

        hardware_interface::CallbackReturn
        on_configure(const rclcpp_lifecycle::State &previous_state) override;

        hardware_interface::CallbackReturn
        on_activate(const rclcpp_lifecycle::State &previous_state) override;

        hardware_interface::CallbackReturn
        on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

        hardware_interface::CallbackReturn
        on_cleanup(const rclcpp_lifecycle::State &previous_state) override;

        std::vector<hardware_interface::StateInterface>
        export_state_interfaces() override;

        std::vector<hardware_interface::CommandInterface>
        export_command_interfaces() override;

        hardware_interface::return_type
        read(const rclcpp::Time &time, const rclcpp::Duration &period) override;

        hardware_interface::return_type
        write(const rclcpp::Time &time, const rclcpp::Duration &period) override;

    private:
        SMS_STS driver_;
        std::string port_;
        int baudrate_;
        size_t num_of_joints_;
        uint8_t rx_packet_[4];

        std::vector<double> hw_positions_;
        std::vector<double> hw_velocities_;
        std::vector<double> cmd_positions_;

        std::vector<uint8_t> ids_;
        std::vector<s16> positions_;
        std::vector<u16> velocities_;
        std::vector<u8> accelerations_;

        std::vector<s16> activate_deactivate_positions_;
        std::vector<u16> activate_deactivate_velocities_;
        std::vector<u8> activate_deactivate_accelerations_;

    }; // class SP7HardwareInterface

} // namespace sp7_hardware

#endif // SP7_HARDWARE_INTERFACE_HPP