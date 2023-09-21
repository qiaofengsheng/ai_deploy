mkdir -p build
cd build
cmake ..
make -j 16

export LD_LIBRARY_PATH=/home/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/lib:/home/qfs/package_manager/TensorRT-8.4.1.5/lib/:$LD_LIBRARY_PATH
./demo 
