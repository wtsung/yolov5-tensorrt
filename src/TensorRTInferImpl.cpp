#include "TensorRTInferImpl.h"

#include "Logger.h"
#include "DataProcess.h"
#include "TimeUtil.h"

#include <numeric>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include "NvOnnxParser.h"

#define NMS_THRESHOLD 0.65

static std::string tmpFP32TrtEnginePath = "./tmp_yolov5x_fp32.trt";

static bool outputModuleCostTime = false;

static Logger gLogger{nvinfer1::ILogger::Severity::kINFO};

void TensorRTInferImpl::loadModelFromFile(const std::string &onnxModelPath, const std::string &trtEnginePath) {
    // determine if there is an engine serialization file
    std::ifstream file(trtEnginePath);
    if (file.good()) {
        if (!geneTrtEngine(trtEnginePath)) {
            std::cout << "Generate TensorRt Engine failed" << std::endl;
            return ;
        }
    } else {
        std::cout << "Model " << trtEnginePath << " not exist" << std::endl;
        std::cout << "Try load default engine serialization file tmp_yolov5x_fp32.trt" << std::endl;

        std::ifstream file(tmpFP32TrtEnginePath);
        if (!file.good()) {
            std::cout <<
                    "Default engine serialization file not exist, try to generate serialization engine"
                    << std::endl;
            if (!geneSerializationEngine(onnxModelPath)) {
                std::cout << "Generate default serialization engine fail" << std::endl;
                return ;
            }
            std::cout << "Generate default serialization engine success" << std::endl;
        }

        if (!geneTrtEngine(tmpFP32TrtEnginePath)) {
            std::cout << "Generate default tmpFP32TrtEngine failed" << std::endl;
            return ;
        }
    }

    // create an IExecutionContext object to manage the inference context
    mExecContext.reset(mCudaEngine->createExecutionContext());

    // Get model input tensor information
    auto inputTensorName = mCudaEngine->getIOTensorName(0);
    nvinfer1::Dims inputTensorDims = mCudaEngine->getTensorShape(inputTensorName);
    mInputImageBatch = inputTensorDims.d[0];
    mInputImageChannel = inputTensorDims.d[1];
    mInputImageHeight = inputTensorDims.d[2];
    mInputImageWidth = inputTensorDims.d[3];
    std::cout << "Model Input Tensor Shape: Batch=" << mInputImageBatch << ", Channel=" << mInputImageChannel
            << ", Height=" << mInputImageHeight << ", Width=" << mInputImageWidth << std::endl;

    // Model warm-up. The first inference after directly deserializing the engine takes a long time
    // and requires preparation work such as opening up gpu memory and exchanging data space.
    modelInferenceWarmup();
}

void TensorRTInferImpl::prepareImageGpu(cv::Mat &inputImage, float *modelInputGpuMem) {
    cv::cuda::GpuMat devInputImage;
    devInputImage.upload(inputImage);

    mRatio = std::min(mInputImageWidth / (inputImage.cols * 1.0f), mInputImageHeight / (inputImage.rows * 1.0f));

    /*
     * input image 1920 * 1080
     * mRatio = 1/3
     * (scaleWidth, scaleHeight) = (640, 360)
     * (mXOffset, mYOffset) = (0, 140)
     */
    // scale
    const int scaleWidth = inputImage.cols * mRatio;
    const int scaleHeight = inputImage.rows * mRatio;

    cv::cuda::GpuMat devResizedImage;
    cv::cuda::resize(devInputImage, devResizedImage, cv::Size(scaleWidth, scaleHeight));

    // padding offset
    mXOffset = (mInputImageWidth - scaleWidth) / 2;
    mYOffset = (mInputImageHeight - scaleHeight) / 2;
    // expand the image boundaries to meet the model input
    cv::cuda::copyMakeBorder(devResizedImage, devResizedImage, mYOffset, mYOffset, mXOffset,
                             mXOffset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // normalization
    cv::cuda::GpuMat devNormalizedImage;
    devResizedImage.convertTo(devNormalizedImage, CV_32FC3, 1.0 / 255.0);

    // bgr to rgb
    int channelLength = mInputImageWidth * mInputImageHeight;
    std::vector<cv::cuda::GpuMat> devSplitImg = {
        cv::cuda::GpuMat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputGpuMem + channelLength * 2),
        cv::cuda::GpuMat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputGpuMem + channelLength * 1),
        cv::cuda::GpuMat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputGpuMem + channelLength * 0)
    };
    cv::cuda::split(devNormalizedImage, devSplitImg);
}

void TensorRTInferImpl::prepareImageCpu(cv::Mat &inputImage, float *modelInputCpuMem) {
    cv::Mat resizedImage;

    mRatio = std::min(mInputImageWidth / (inputImage.cols * 1.0f), mInputImageHeight / (inputImage.rows * 1.0f));

    /*
     * input image 1920 * 1080
     * mRatio = 1/3
     * (scaleWidth, scaleHeight) = (640, 360)
     * (mXOffset, mYOffset) = (0, 140)
     */
    // scale
    const int scaleWidth = inputImage.cols * mRatio;
    const int scaleHeight = inputImage.rows * mRatio;
    cv::resize(inputImage, resizedImage, cv::Size(scaleWidth, scaleHeight));


    // padding offset
    mXOffset = (mInputImageWidth - scaleWidth) / 2;
    mYOffset = (mInputImageHeight - scaleHeight) / 2;
    // 扩展图像边界，满足模型输入，填充值为X/Y方向的offset
    cv::copyMakeBorder(resizedImage, resizedImage, mYOffset, mYOffset, mXOffset,
                       mXOffset, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // normalization
    cv::Mat normalizedImage;
    resizedImage.convertTo(normalizedImage, CV_32FC3, 1.0 / 255.0);

    // bgr to rgb
    int channelLength = mInputImageWidth * mInputImageHeight;
    std::vector<cv::Mat> split_img = {
        cv::Mat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputCpuMem + channelLength * 2),
        cv::Mat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputCpuMem + channelLength * 1),
        cv::Mat(mInputImageHeight, mInputImageWidth, CV_32FC1, modelInputCpuMem + channelLength * 0)
    };
    cv::split(normalizedImage, split_img);
}

void TensorRTInferImpl::postProcessCpu(float *inFeatureMap, std::vector<DetectedObject> &outDetObjects) {
    std::vector<DetectedObject> objs;
    yoloBoxCpu(inFeatureMap, objs, mXOffset, mYOffset, mRatio);
    if (!objs.empty()) {
        nmsCpu(objs, outDetObjects, NMS_THRESHOLD);
    }
}

void TensorRTInferImpl::postProcessGpu(void *inFeatureMap, std::vector<DetectedObject> &outDetObjects) {
    DetectedObject *devDetObjMem = nullptr;
    int *devAtomicDetObjIdx = nullptr;
    yoloBoxGpu(static_cast<float *>(inFeatureMap), &devDetObjMem, &devAtomicDetObjIdx, mXOffset, mYOffset, mRatio);

    nmsGpu(&devDetObjMem, &devAtomicDetObjIdx, outDetObjects, NMS_THRESHOLD);

    cudaFree(devDetObjMem);
    cudaFree(devAtomicDetObjIdx);
}

inline unsigned int GetElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32: {
            return 4;
        }
        case nvinfer1::DataType::kFLOAT: {
            return 4;
        }
        case nvinfer1::DataType::kHALF: {
            return 2;
        }
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: {
            return 1;
        }
        default: {
            throw std::runtime_error("Invalid DataType.");
        }
    }
    return 0;
}

inline int64_t Volume(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>()); //累乘
}

bool TensorRTInferImpl::inference(cv::Mat &inputImage, std::vector<DetectedObject> &outDetObjects) {
    outDetObjects.clear();

    if (!mCudaEngine) {
        std::cout << "engine is null" << std::endl;
        return false;
    }

    if (!mExecContext) {
        std::cout << "context is null" << std::endl;
        return false;
    }

    void *buffers[2];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // get model input tensot size and allocate device memory
    auto inputTensorName = mCudaEngine->getIOTensorName(0);
    nvinfer1::Dims inputTensorDims = mCudaEngine->getTensorShape(inputTensorName);
    int64_t inputTensorSize = Volume(inputTensorDims) * GetElementSize(mCudaEngine->getTensorDataType(inputTensorName));
    cudaMallocAsync(&buffers[0], inputTensorSize, stream);

    // Get the model output tensor size and allocate device memory
    auto outputTensorName = mCudaEngine->getIOTensorName(1);
    nvinfer1::Dims outputTensorDims = mCudaEngine->getTensorShape(outputTensorName);
    int64_t outputTensorSize = Volume(outputTensorDims) * GetElementSize(
                                   mCudaEngine->getTensorDataType(outputTensorName));
    cudaMallocAsync(&buffers[1], outputTensorSize, stream);

    TimeUtil timer;
    // Image pre-processing
#ifdef USE_OPENCV_GPU
    // prepareImageGpu outputs memory on gpu
    prepareImageGpu(inputImage, static_cast<float *>(buffers[0]));
#else
    std::vector<float> preparedInputImgaeCpu(mInputImageBatch * mInputImageWidth * mInputImageHeight * mInputImageChannel);
    prepareImageCpu(inputImage, preparedInputImgaeCpu.data());
    // Copy the host-side data obtained by image pre-processing to the gpu memory
    cudaMemcpyAsync(buffers[0], preparedInputImgaeCpu.data(), inputTensorSize, cudaMemcpyHostToDevice, stream);
#endif
    if (outputModuleCostTime) {
        std::cout << "Pre-process cost: " << timer.duration() << " ms." << std::endl;
    }


    timer.restart();
    // model Inference
    mExecContext->enqueueV2(buffers, stream, nullptr); // async
    //mExecContext->executeV2(buffers);// sync
    if (outputModuleCostTime) {
        std::cout << "Context-inference cost: " << timer.duration() << " ms." << std::endl;
    }


    timer.restart();
    // post-processing for model output featureMap
#ifdef USE_POST_PROCESS_GPU
    cudaStreamSynchronize(stream);
    // model output data on the device side is processed directly on the device side
    postProcessGpu(buffers[1], outDetObjects); //todo 传入stream
#else
    // allocate host memory to the size of the model output data
    std::vector<float> hostOutputFeatureMap(Volume(outputTensorDims));
    // copy the model output data from the Device to the host
    cudaMemcpyAsync(hostOutputFeatureMap.data(), buffers[1], outputTensorSize, cudaMemcpyDeviceToHost, stream);
    // wait for cudaStream
    cudaStreamSynchronize(stream);
    // post-processing in cpu
    postProcessCpu(hostOutputFeatureMap.data(), outDetObjects);
#endif
    if (outputModuleCostTime) {
        std::cout << "Post-process cost: " << timer.duration() << " ms." << std::endl;
    }

    cudaFreeAsync(buffers[0], stream);
    cudaFreeAsync(buffers[1], stream);
    cudaStreamDestroy(stream);

    return true;
}

void TensorRTInferImpl::modelInferenceWarmup(int warmupCount) {
    std::vector<DetectedObject> outDetObjects;
    // do not use images of the same size as the model input tensor
    // GPU pre-processing also requires warmup
    cv::Mat warmupImg = cv::Mat::zeros(1920, 1080, CV_32FC3);
    for (int i = 0; i < warmupCount; ++i) {
        inference(warmupImg, outDetObjects);
    }

    // enable costTime output after warmup
    outputModuleCostTime = true;
}

bool TensorRTInferImpl::geneSerializationEngine(const std::string &onnxModelPath) {
    // the default conversion engine uses fp32 precision.
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);

    // construct model network structure
    // explicitBatch=0, implicit batch, the default batch input of the model,
    // the rest are explicit batches, here is 1, explicit can be adjusted during inference
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(
                                       nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    // create the ONNX parser, load the model into the network after parsing,
    // build the Network calculation graph
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

    // get the analysis error output information
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
        return false;
    }

    // create an IBuilderConfig object to control the optimization of the model
    // setMemoryPoolLimit limits the maximum size that can be used by any layer in the network
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 25);

    // building the serializedNetwork
    nvinfer1::IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);

    // serialize the model into the engine file
    std::stringstream engineFileStream;
    engineFileStream.seekg(0, std::stringstream::beg);
    engineFileStream.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());

    std::ofstream out_file(tmpFP32TrtEnginePath);
    assert(out_file.is_open());
    out_file << engineFileStream.rdbuf();
    out_file.close();
    std::cout << "trt model serialize save as " << tmpFP32TrtEnginePath << std::endl;

    delete config;
    delete parser;
    delete network;
    delete builder;
    delete serializedModel;

    return true;
}

bool TensorRTInferImpl::geneTrtEngine(const std::string &trtEnginePath) {
    // the engine file exists, just deserialize the engine file
    std::cout << "Load model " << trtEnginePath << std::endl;

    std::fstream fileStream;
    fileStream.open(trtEnginePath, std::ios::binary | std::ios::in);

    if (!fileStream.is_open()) {
        std::cout << "Engine file open failed" << std::endl;
        return false;
    }

    std::stringstream engineBuffer;
    engineBuffer << fileStream.rdbuf();

    std::string cachedEngine;
    cachedEngine.append(engineBuffer.str());

    fileStream.close();

    // IRuntime is used to deserialize an engine
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

    // The engine is deserialized from the stream
    mCudaEngine.reset(runtime->deserializeCudaEngine(cachedEngine.data(), cachedEngine.size()));

    std::cout << "trt model deserialize done" << std::endl;
    return true;
}
