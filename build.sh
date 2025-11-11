#!/bin/bash
set  -e 

if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
echo "Build completed. Executable and reports are in the build/ directory."