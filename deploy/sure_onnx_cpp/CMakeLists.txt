cmake_minimum_required(VERSION 3.0)
project(sure_onnx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# onnxruntime providers, default use CPU
set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.18.0")

# option(onnxruntime_USE_CUDA "Build with CUDA support" ON) 
# set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-gpu-1.18.0")

# option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)

message("onnxruntime ${ONNXRUNTIME_ROOTDIR}")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

find_package(OpenCV REQUIRED)

add_library(sure ./sure/sure.cpp)
target_link_libraries(sure ${OpenCV_LIBRARIES} onnxruntime)

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBRARIES} onnxruntime sure)