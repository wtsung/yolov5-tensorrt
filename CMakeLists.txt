cmake_minimum_required(VERSION 3.22)
project(yolo_v5_tensorrt LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-DUSE_OPENCV_GPU)
set(USE_OPENCV_GPU 1)

add_definitions(-DUSE_POST_PROCESS_GPU)
set(USE_POST_PROCESS_GPU 1)

add_definitions(-DUSE_POST_PROCESS_GPU)
set(USE_POST_PROCESS_GPU 1)


find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third/opencv-4.8.0/include/opencv4)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third/opencv-4.8.0/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third/TensorRT-8.5.1.7/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third/TensorRT-8.5.1.7/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE(yolo_v5_tensorrt main.cpp
        src/Logger.h
        src/TensorRTInferImpl.h
        src/BaseDef.h
        src/TensorRTInferImpl.cpp
        src/DataProcess.h
        src/DataProcess.cu
        src/NNInfer.h
        src/TimeUtil.h
        src/BaseDef.cpp)

target_link_libraries(yolo_v5_tensorrt ${CUDA_LIBRARIES} nvinfer nvonnxparser opencv_cudawarping opencv_cudaarithm opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn)