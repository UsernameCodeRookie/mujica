cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Mujica VERSION 1.0)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the directory for header files
include_directories(include)

# Add the source file
add_executable(mujica src/main.cpp)