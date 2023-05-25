project(demo)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
include_directories(
    /data/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/include
    /data/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/include
    /data/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/include
    /usr/local/cuda/include
    ./include
    /data/qfs/package_manager/qfs_3rdParty/json/include
)

file(GLOB LIBS_FILE
    /data/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/lib/*.so
    /data/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/lib/*
    /usr/local/cuda/lib64/*
    /data/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/lib/*
    /data/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/static_lib/*
    /data/qfs/package_manager/qfs_3rdParty/json/lib/*.so
)

add_executable(demo main.cpp src/external_interface.cpp src/frame_state_inference.cpp src/tensorrt.cpp)
# add_library(demo SHARED src/external_interface.cpp src/frame_state_detection.cpp src/tensorrt.cpp)
target_link_libraries(demo ${LIBS_FILE} -lpthread)