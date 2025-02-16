cmake_minimum_required(VERSION 3.10)

# Project name and version
project(aBunchOfSlam VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Specify the required libraries
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# Define the include directories for the libraries
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    /usr/local/include
)

# Manually specify which source files should be compiled into executables
set(SOURCE_FILES
    src/icp_svd.cpp
    src/icp_gnm_point_to_point.cpp
)

# Set the directory for the executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/scripts)

# Create an executable for each specified source file
foreach(SRC_FILE ${SOURCE_FILES})
    get_filename_component(EXEC_NAME ${SRC_FILE} NAME_WE)
    add_executable(${EXEC_NAME} ${SRC_FILE} src/eigen.cpp src/pangolin.cpp)
    target_link_libraries(${EXEC_NAME}
        ${Python3_LIBRARIES}
        ${Pangolin_LIBRARIES}
    )
endforeach()
