
### Project Introduction
This project focuses on optimizing the inference performance of the YOLOv5x model using TensorRT.

#### Directory
- images: Contains test image and model output results.
- models: Houses the TensorRT engine serialization files with INT8 precision, and scripts for model conversion.
- src: Contains the source code for implementation.
- third: Includes TensorRT and OpenCV libraries.

#### Environment

##### Base
**GPU:** NVIDIA GeForce RTX 4060  \
**docker:** Use the NVIDIA CUDA container: nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

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

The YOLOv5x ONNX model can be obtained from [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5/blob/master/export.py) using the following command:

>python export.py --weights yolov5x.pt --include onnx --imgsz 640 640


#### Calibration Dataset
The calibration dataset can be downloaded from: [COCO 2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip).

#### Third Party Library

The project needs to link two libraries (Note: The library file in the current third directory is empty):
> opencv-4.8.0 with CUDA \
> TensorRT-8.5.1.7

Download links for the above libraries are provided in /third/download_paths.txt
