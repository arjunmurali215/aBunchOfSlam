cmake_minimum_required(VERSION 3.10)

# Project name and version
project(aBunchOfSlam VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

# Specify the required libraries
find_package(Eigen3 REQUIRED)
<<<<<<< HEAD
find_package(Pangolin REQUIRED)  # Add Pangolin

# Define the source files
set(SOURCES
    src/icp_svd.cpp  # Your file name
)
=======
find_package(Pangolin REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
>>>>>>> 68c6e87 (changed CMakeLists.txt)

# Define the include directories for the libraries
include_directories(
    ${EIGEN3_INCLUDE_DIR}
<<<<<<< HEAD
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
=======
    ${Pangolin_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    /usr/local/include
)

# Find all source files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Set the directory for the executables in the project directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/scripts)

# Create an executable for each source file
foreach(SRC_FILE ${SOURCES})
    get_filename_component(EXEC_NAME ${SRC_FILE} NAME_WE)
    add_executable(${EXEC_NAME} ${SRC_FILE})
    target_link_libraries(${EXEC_NAME}
        ${Python3_LIBRARIES}
        ${Pangolin_LIBRARIES}
    )
endforeach()
>>>>>>> 68c6e87 (changed CMakeLists.txt)
