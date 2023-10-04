cmake_minimum_required(VERSION 3.2.2)
project(calib)

# BUILD TYPE
set(CMAKE_BUILD_TYPE Release)

# FIND BOOST
find_package(Ceres)
find_package(Eigen3 REQUIRED)


# Demonstration of the UGPM
add_executable(imu_pose_calib app/imu_pose_calib.cpp)
set_target_properties(imu_pose_calib PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories( imu_pose_calib PUBLIC
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    )

target_link_libraries( imu_pose_calib PUBLIC
    ${CERES_LIBRARIES}
    )





