cmake_minimum_required(VERSION 3.5)

project(ransac_solver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Find Eigen3. If it defines the target, this is used. If not,
# fall back to the using the module form.
# See https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html for details
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found.")
else()
  message(FATAL_ERROR "Eigen3 > 3.1.0 not found.")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found.")
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# Find gflags
find_package(gflags REQUIRED)
if (gflags_FOUND)
    include_directories(${gflags_INCLUDE_DIRS})
    message(STATUS "gflags found.")
else()
    message(FATAL_ERROR "gflags not found.")
endif()

include_directories(./include)
include_directories(./examples)

add_executable(calc_essential_matrix ./examples/calc_essential_matrix.cpp ./examples/reconstruction.cpp)
target_link_libraries(calc_essential_matrix ${OpenCV_LIBS} gflags)

