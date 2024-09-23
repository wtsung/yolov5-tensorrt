#ifndef TENSORRTINFERIMPL_H
#define TENSORRTINFERIMPL_H

#include "NvInferRuntime.h"

#include "NNInfer.h"

class TensorRTInferImpl : public NNInfer {
public:
    void loadModelFromFile(const std::string &onnxModelPath, const std::string &trtEnginePath) override;

    bool inference(cv::Mat &inputImage, std::vector<DetectedObject> &detObjects) override;

private:
    void prepareImageCpu(cv::Mat &inputImage, float *modelInputCpuMem);

    void prepareImageGpu(cv::Mat &inputImage, float *modelInputGpuMem);

    void postProcessCpu(float *inFeatureMap, std::vector<DetectedObject> &outDetObjects);

    void postProcessGpu(void *inFeatureMap, std::vector<DetectedObject> &outDetObjects);

    void modelInferenceWarmup(int warmupCount = 10);

    bool geneSerializationEngine(const std::string &onnxModelPath);

    bool geneTrtEngine(const std::string &trtEnginePath);

    // ICudaEngine is the inference engine in TensorRT, which contains the optimized network model
    // This model is optimized and serialized by TensorRT
    std::unique_ptr<nvinfer1::ICudaEngine> mCudaEngine = nullptr;

    // model inference context created by ICudaEngine
    // Multiple execution contexts can exist in a single ICudaEngine instance,
    // allowing multiple batches of inference to be executed simultaneously using the same engine.
    std::unique_ptr<nvinfer1::IExecutionContext> mExecContext = nullptr;

    // model input tensor N W H C
    int mInputImageBatch = 0;
    int mInputImageWidth = 0;
    int mInputImageHeight = 0;
    int mInputImageChannel = 0;

    int mXOffset = 0;
    int mYOffset = 0;
    float mRatio = 0.0;
};

#endif //TENSORRTINFERIMPL_H
