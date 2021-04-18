cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(host)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(host host.cpp)
target_link_libraries(host "${TORCH_LIBRARIES}")
set_property(TARGET host PROPERTY CXX_STANDARD 14)

# Torch location: /home/rubin3737/libtorch