#ifndef NNINFER_H
#define NNINFER_H

#include <string>
#include <opencv2/opencv.hpp>

#include "BaseDef.h"

class NNInfer {
public:
    /**
     * @brief Loading model files
     *
     * Prioritize loading heterogeneous format model files，
     * If it does not exist, load the onnx format model file, convert the format and save it
     *
     * @param[in] onnxModelPath  Model ONNX format file path
     * @param[in] heterModelPath Model heterogeneous format file path
     *
     * @return
     */
    virtual void loadModelFromFile(const std::string &onnxModelPath, const std::string &heterModelPath) = 0;

    /**
     * @brief Input the image to perform model inference and return result
     *
     *
     * @param[in]  inputImage  Input image
     * @param[out] detObjects  Model output results
     *
     * @return Inference interface call result
     */
    virtual bool inference(cv::Mat &inputImage, std::vector<DetectedObject> &detObjects) = 0;
};

#endif //NNINFER_H
