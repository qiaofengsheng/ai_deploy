project(demo)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
include_directories(
    /home/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/include
    /home/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/include
    /home/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/include
    /usr/local/cuda/include
    ./include
    /home/qfs/package_manager/qfs_3rdParty/json/include
)

file(GLOB LIBS_FILE
    /home/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/lib/*.so
    /home/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/lib/*
    /usr/local/cuda/lib64/*
    /home/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/lib/*
    /home/qfs/package_manager/qfs_3rdParty/boost/linux/boost_1_65_1/static_lib/*
    /home/qfs/package_manager/qfs_3rdParty/json/lib/*.so
)

file(GLOB files_
    src/*.cpp
)

add_library(unet_seg SHARED ${files_})
add_executable(demo main.cpp ${files_})

target_link_libraries(unet_seg ${LIBS_FILE} -lpthread)
target_link_libraries(demo ${LIBS_FILE} -lpthread)