from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    launch_arguments = [
    ]

    description_pkg = FindPackageShare('sp7_description')
    bringup_pkg = FindPackageShare('sp7_bringup')
    xacro_path = PathJoinSubstitution([description_pkg, 'urdf', 'sp7.urdf.xacro'])
    sp7_controllers = PathJoinSubstitution([bringup_pkg, 'config', 'sp7_controllers.yaml'])

    robot_description = ParameterValue(Command([
        'xacro', ' ', xacro_path, ' ',
    ]), value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description}],
        output="screen"
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[sp7_controllers],
        output="screen"
    )

    controllers_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=['joint_state_broadcaster', 'sp7_controller', '--param-file', sp7_controllers],
        output="screen"
    )

    return LaunchDescription([
        *launch_arguments,
        robot_state_publisher_node,
        control_node,
        controllers_spawner,
    ])
