cmake_minimum_required(VERSION 3.20)
project(sp7_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(st_sc_servo_control_lib REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/sp7_hardware.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC include)

ament_target_dependencies(${PROJECT_NAME}
rclcpp
rclcpp_lifecycle
hardware_interface
pluginlib
st_sc_servo_control_lib
)

pluginlib_export_plugin_description_file(hardware_interface sp7_hardware.xml)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

ament_package()
