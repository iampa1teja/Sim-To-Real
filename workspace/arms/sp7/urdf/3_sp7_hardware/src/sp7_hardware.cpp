#include "sp7_hardware/sp7_hardware.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace sp7_hardware
{
    hardware_interface::CallbackReturn SP7HardwareInterface::on_init(const hardware_interface::HardwareComponentInterfaceParams &params)
    {
        if (hardware_interface::SystemInterface::on_init(params) != hardware_interface::CallbackReturn::SUCCESS)
        {
            return hardware_interface::CallbackReturn::ERROR;
        }

        port_ = params.hardware_info.hardware_parameters["port"];
        baudrate_ = 115200;
        joints_ = params.hardware_info.joints;

        hw_positions_.resize(joints_.size(), 0.0);
        hw_velocities_.resize(joints_.size(), 0.0);
        cmd_positions_.resize(joints_.size(), 0.0);
        ids_.resize(joints_.size());

        for (size_t i = 0; i < joints_.size(); i++)
        {
            ids_[i] = std::stoi(joints_[i].parameters.at("id"));
        }

        positions_.resize(joints_.size(), 2048);  // range: 0-4095
        velocities_.resize(joints_.size(), 1365); // range: 0-3072
        accelerations_.resize(joints_.size(), 5); // range: 0-150

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> SP7HardwareInterface::export_state_interfaces()
    {
        std::vector<hardware_interface::StateInterface> state_interfaces;

        for (size_t i = 0; i < joints_.size(); i++)
        {
            state_interfaces.emplace_back(hardware_interface::StateInterface(joints_[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(joints_[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
        }
        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> SP7HardwareInterface::export_command_interfaces()
    {
        std::vector<hardware_interface::CommandInterface> command_interfaces;

        for (size_t i = 0; i < joints_.size(); i++)
        {
            command_interfaces.emplace_back(hardware_interface::CommandInterface(joints_[i].name, hardware_interface::HW_IF_POSITION, &cmd_positions_[i]));
        }
        return command_interfaces;
    }

    hardware_interface::CallbackReturn SP7HardwareInterface::on_configure(const rclcpp_lifecycle::State &)
    {
        if (!driver_.begin(baudrate_, port_.data()))
        {
            return hardware_interface::CallbackReturn::ERROR;
        }

        usleep(SLEEP_TIME_AFTER_BEGIN);

        driver_.syncReadBegin(joints_.size(), sizeof(rx_packet_));

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn SP7HardwareInterface::on_activate(const rclcpp_lifecycle::State &)
    {
        for (size_t i = 0; i < joints_.size(); i++)
        {
            driver_.EnableTorque(static_cast<int>(ids_[i]), 1);
            usleep(SLEEP_TIME_TOGGLE_TORQUE);
        }

        driver_.SyncWritePosEx(
            ids_.data(),
            joints_.size(),
            activate_deactivate_positions_.data(),
            activate_deactivate_velocities_.data(),
            activate_deactivate_accelerations_.data());

        usleep(SLEEP_TIME_ACTIVATE_DEACTIVATE);

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn SP7HardwareInterface::on_deactivate(const rclcpp_lifecycle::State &)
    {
        driver_.SyncWritePosEx(
            ids_.data(),
            joints_.size(),
            activate_deactivate_positions_.data(),
            activate_deactivate_velocities_.data(),
            activate_deactivate_accelerations_.data());

        usleep(SLEEP_TIME_ACTIVATE_DEACTIVATE);

        for (size_t i = 0; i < joints_.size(); i++)
        {
            driver_.EnableTorque(static_cast<int>(ids_[i]), 0);
            usleep(SLEEP_TIME_TOGGLE_TORQUE);
        }

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn SP7HardwareInterface::on_cleanup(const rclcpp_lifecycle::State &)
    {
        driver_.syncReadEnd();

        driver_.end();

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type SP7HardwareInterface::read(const rclcpp::Time &, const rclcpp::Duration &)
    {
        driver_.syncReadPacketTx(
            ids_.data(),
            joints_.size(),
            SMS_STS_PRESENT_POSITION_L,
            sizeof(rx_packet_));

        for (size_t i = 0; i < joints_.size(); i++)
        {
            if (!driver_.syncReadPacketRx(ids_[i], rx_packet_))
            {
                continue;
            }
            int16_t raw_position = driver_.syncReadRxPacketToWrod(15);
            int16_t raw_speed = driver_.syncReadRxPacketToWrod(15);

            hw_positions_[i] = RAW_POS_TO_RAD(raw_position);
            hw_velocities_[i] = RAW_SPEED_TO_RAD_PER_SEC(raw_speed);
        }

        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type SP7HardwareInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
    {
        for (size_t i = 0; i < joints_.size(); i++)
        {
            s16 target_pos = static_cast<s16>((cmd_positions_[i] / M_PI / 2.0 + 0.5) * 4096.0); // TODO: Macro
            positions_[i] = std::clamp(target_pos, 0, 4095);
        }

        driver_.SyncWritePosEx(
            ids_.data(),
            joints_.size(),
            positions_.data(),
            velocities_.data(),
            accelerations_.data());

        return hardware_interface::return_type::OK;
    }

} // namespace sp7_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    sp7_hardware::SP7HardwareInterface,
    hardware_interface::SystemInterface)