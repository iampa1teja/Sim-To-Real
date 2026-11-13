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
    xacro_path = PathJoinSubstitution([description_pkg, 'urdf', 'sp7.urdf.xacro'])
    # rviz_config_path = PathJoinSubstitution([description_pkg, 'rviz', 'display.rviz'])

    robot_description = ParameterValue(Command([
        'xacro', ' ', xacro_path, ' ',
    ]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description}],
        output="screen"
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        # arguments=['-d', rviz_config_path],
        output="screen"
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen"
    )

    return LaunchDescription([
        *launch_arguments,
        robot_state_publisher,
        rviz2,
        joint_state_publisher_gui,
    ])
