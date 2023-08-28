mkdir -p build
cd build
cmake ..
make -j 16

export LD_LIBRARY_PATH=/home/qfs/package_manager/qfs_3rdParty/opencv/linux/4.2.0/lib:$LD_LIBRARY_PATH
./demo 
