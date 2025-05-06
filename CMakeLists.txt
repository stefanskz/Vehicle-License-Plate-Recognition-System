cmake_minimum_required(VERSION 3.29)
project(project2.0)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(project2.0 main.cpp)

target_link_libraries(project2.0 ${OpenCV_LIBS})