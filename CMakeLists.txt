cmake_minimum_required(VERSION 3.10)

# Project name and version
project(aBunchOfSlam VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Specify the required libraries
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)  # Add Pangolin

# Define the source files
set(SOURCES
    src/icp_svd.cpp  # Your file name
)

# Define the include directories for the libraries
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}  # Include Pangolin directories
    /usr/local/include
)

# Set the executable output path
add_executable(${PROJECT_NAME} ${SOURCES})

# If you are using Python for Matplotlib-C++, make sure Python is found
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${Python3_LIBRARIES}
    ${Pangolin_LIBRARIES}  # Link Pangolin
)
