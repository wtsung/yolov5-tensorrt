
### Project Introduction
Deploy the yolov5x model using TensorRT.

#### Directory
- images: Test image and model output result image.
- models: yolov5x onnx model，tensorRT engine serialization file，Including FP32 precision and INT8 precision.
- src: Source code.
- third: tensorRT and opencv.

#### Environment

##### Base
**GPU:** NVIDIA GeForce RTX 4060  \
**镜像:** nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

##### Container Configuration
###### Install basic software
>**python3:** python_version >= 3.8 \
>**pip3:** apt install python3-pip \
>**numpy:** pip3 install numpy==1.23.2 \
>**pycuda:** pip3 install pycuda \
>**tensorrt:** pip3 install nvidia-pyindex && pip3 install tensorrt==8.5.1.7 \
>**opencv:** apt install libgl1-mesa-dev, pip3 install opencv-python  

###### Environment variables
>export PATH=$PATH:/usr/local/cuda/bin

#### Model

Model yolov5x.onnx obtained through https://github.com/ultralytics/yolov5/blob/master/export.py

>python export.py --weights yolov5x.pt --include onnx --imgsz 640 640


Calibration Dataset：http://images.cocodataset.org/zips/val2017.zip


#### Third Party Library

The project needs to link two libraries (Note: The library file in the current third directory is empty):
> opencv-4.8.0 with CUDA \
> TensorRT-8.5.1.7

Download links for the above libraries are provided in /third/download_paths.txt