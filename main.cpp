#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "TensorRTInferImpl.h"
#include "NvOnnxParser.h"

#include "BaseDef.h"
#include "TimeUtil.h"

#define IMAGE_PATH        "/home/yolo_v5_tensorrt/images/image.jpg"
#define RESULT_IMAGE_PATH "/home/yolo_v5_tensorrt/images/result.jpg"
#define ONNX_MODEL_PATH   "/home/yolo_v5_tensorrt/models/yolov5x.onnx"
#define TRT_ENGINE_INT8_PATH   "/home/yolo_v5_tensorrt/models/yolov5x_int8.trt"
#define TRT_ENGINE_FP32_PATH   "/home/yolo_v5_tensorrt/models/yolov5x_fp32.trt"

int main() {
    std::cout << "===================" << " Start Inference Network " << "==================" << std::endl;

    std::shared_ptr<NNInfer> infer = std::make_shared<TensorRTInferImpl>();

    infer->loadModelFromFile(ONNX_MODEL_PATH, TRT_ENGINE_INT8_PATH);

    cv::Mat inputImageMat = cv::imread(IMAGE_PATH);

    TimeUtil timer;

    std::vector<DetectedObject> detObjects;
    infer->inference(inputImageMat, detObjects);

    std::cout << "Inference cost: " << timer.duration() << " ms." << std::endl;

    // write detect result to origin image
    for (const auto &detObject: detObjects) {
        std::cout << detObject.box.x << ", " << detObject.box.y << ", " << detObject.box.width << ", " << detObject.box.
                height << ", className=" << getClassName(detObject.classId) << std::endl;
        const cv::Point topLeft(detObject.box.x, detObject.box.y);
        const cv::Point bottomRight(detObject.box.x + detObject.box.width, detObject.box.y + detObject.box.height);
        cv::rectangle(inputImageMat, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);
    }
    cv::imwrite(RESULT_IMAGE_PATH, inputImageMat);

    return 0;
}
