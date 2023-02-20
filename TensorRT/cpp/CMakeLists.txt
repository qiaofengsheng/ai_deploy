project(demo)
cmake_minimum_required(VERSION 3.10)

include_directories(
    /data/qfs/project/study/trt_demo/c++/include
    /data/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/include/
    # /data/qfs/package_manager/qfs_3rdParty/cuda/cuda-10.2/include/
    /data/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/include/
    /usr/local/cuda/include/
)

file(GLOB LIBS_PATH
    /data/qfs/package_manager/qfs_3rdParty/tensorrt/linux/cuda-10.2/7.2.2.3/lib/*.so
    # /data/qfs/package_manager/qfs_3rdParty/cuda/cuda-10.2/lib64/*.so
    /usr/local/cuda/lib64/*
    /data/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/lib/*.so

)

add_executable(demo main.cpp trtInference.cpp src/logger.cpp)
# add_executable(demo infer.cpp src/logger.cpp)
# add_executable(demo main.cpp)
target_link_libraries(demo ${LIBS_PATH})