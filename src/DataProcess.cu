#include "DataProcess.h"

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__host__ static float calculateIou(const ObjectBox &lhs, const ObjectBox &rhs) {
    // Coordinates of the upper left point of the intersection area
    float maxX = std::max(lhs.x, rhs.x);
    float maxY = std::max(lhs.y, rhs.y);

    // Coordinates of the lower right point of the intersection area
    float minX = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    float minY = std::min(lhs.y + lhs.height, rhs.y + rhs.height);

    if (minX <= maxX || minY <= maxY) {
        return 0;
    }
    float overArea = (minX - maxX) * (minY - maxY);
    return overArea / (lhs.width * lhs.height + rhs.width * rhs.height - overArea);
}

void nmsCpu(std::vector<DetectedObject> &inDetBoxes, std::vector<DetectedObject> &outNMSBoxs, float nmsThreshold) {
    std::sort(inDetBoxes.begin(), inDetBoxes.end(), [](const DetectedObject &lhs, const DetectedObject &rhs) {
        return lhs.conf > rhs.conf;
    });

    std::vector<bool> isSuppressed(inDetBoxes.size(), false);

    for (size_t i = 0; i < inDetBoxes.size(); ++i) {
        if (isSuppressed[i]) {
            continue;
        }

        // save the current maximum confidence box
        outNMSBoxs.push_back(inDetBoxes[i]);

        for (size_t j = i + 1; j < inDetBoxes.size(); ++j) {
            if (isSuppressed[j]) {
                continue;
            }
            if (inDetBoxes[i].classId == inDetBoxes[j].classId) {
                float iou = calculateIou(inDetBoxes[i].box, inDetBoxes[j].box);
                if (iou > nmsThreshold) {
                    isSuppressed[j] = true;
                }
            }
        }
    }

    std::cout << "NMSCpu inputDetBoxes size=" << inDetBoxes.size() << ", " << "outputNMSBoxs size=" << outNMSBoxs.size()
            << std::endl;
}

__device__ static float calculateIouGpu(const ObjectBox &lhs, const ObjectBox &rhs) {
    // Coordinates of the upper left point of the intersection area
    float maxX = lhs.x > rhs.x ? lhs.x : rhs.x;
    float maxY = lhs.y > rhs.y ? lhs.y : rhs.y;

    // Coordinates of the lower right point of the intersection area
    float minX = lhs.x + lhs.width < rhs.x + rhs.width ? lhs.x + lhs.width : rhs.x + rhs.width;
    float minY = lhs.y + lhs.height < rhs.y + rhs.height ? lhs.y + lhs.height : rhs.y + rhs.height;

    if (minX <= maxX || minY <= maxY) {
        return 0.0;
    }
    float overArea = (minX - maxX) * (minY - maxY);
    return overArea / (lhs.width * lhs.height + rhs.width * rhs.height - overArea);
}

__global__ void nmsGpuKernel(DetectedObject *inSortDetObjs, int *outNmsBoxsFlag, float nmsThreshold, int detBoxesNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= (detBoxesNum * (detBoxesNum - 1))) {
        return;
    }

    int curBoxIdx = idx / detBoxesNum;
    int compareBoxIdx = idx % detBoxesNum;

    if (compareBoxIdx <= curBoxIdx) {
        return;
    }

    // current box only calculates the intersection and union with the box with a larger Idx.
    if (inSortDetObjs[curBoxIdx].classId == inSortDetObjs[compareBoxIdx].classId) {
        float iou = calculateIouGpu(inSortDetObjs[curBoxIdx].box, inSortDetObjs[compareBoxIdx].box);
        if (iou > nmsThreshold) {
            outNmsBoxsFlag[compareBoxIdx] = 0;
        }
    }
}

struct Compare {
    __device__ bool operator()(const DetectedObject &lhs, const DetectedObject &rhs) {
        return lhs.conf > rhs.conf;
    }
};

void nmsGpu(DetectedObject **inDevDetObjMem, int **inDevAtomicDetObjIdx, std::vector<DetectedObject> &outNmsObjs,
            float nmsThreshold) {
    // number of detect objects
    int detObjsNum = 0;
    cudaMemcpy(&detObjsNum, *inDevAtomicDetObjIdx, sizeof(int), cudaMemcpyDeviceToHost);
    if (detObjsNum == 0) {
        return;
    }

    auto devPtrDetObjMem = thrust::device_ptr<DetectedObject>(*inDevDetObjMem);
    thrust::sort(devPtrDetObjMem, devPtrDetObjMem + detObjsNum, Compare());

    // copy the valid flag to gpu
    std::vector<int> hostNmsBoxsFlag(detObjsNum, 1);
    int *devNmsObjsFlag = nullptr;
    cudaMalloc(&devNmsObjsFlag, detObjsNum * sizeof(int));
    cudaMemcpy(devNmsObjsFlag, hostNmsBoxsFlag.data(), detObjsNum * sizeof(int), cudaMemcpyHostToDevice);

    int threadNum = detObjsNum * (detObjsNum - 1);
    const int blockSize = 128;
    const int gridSize = (threadNum + blockSize - 1) / blockSize;
    nmsGpuKernel<<<gridSize, blockSize>>>(*inDevDetObjMem, devNmsObjsFlag, nmsThreshold, detObjsNum);

    // copy the results back to the host
    std::vector<DetectedObject> hostDetObjMem(detObjsNum);
    cudaMemcpy(hostNmsBoxsFlag.data(), devNmsObjsFlag, detObjsNum * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDetObjMem.data(), *inDevDetObjMem, detObjsNum * sizeof(DetectedObject), cudaMemcpyDeviceToHost);

    for (int i = 0; i < detObjsNum; ++i) {
        if (hostNmsBoxsFlag[i] == 1) {
            outNmsObjs.push_back(hostDetObjMem[i]);
        }
    }
    cudaFree(devNmsObjsFlag);

    std::cout << "NMSGpu inputDetBoxes size=" << detObjsNum << ", " << "outputNMSBoxs size=" << outNmsObjs.size() <<
            std::endl;
}

void yoloBoxCpu(const float *inFeatureMap, std::vector<DetectedObject> &outDetObjects, int xOffset, int yOffset,
                float ratio) {
    std::vector<DetectedObject> objs;
    for (int i = 0; i < FEATURE_MAP_BOX_NUM; ++i, inFeatureMap += BOX_FEATURE_MAP_LEN) {
        const float objectConf = inFeatureMap[4];

        if (objectConf < OBJECT_CONF_THRESHOLD) {
            continue;
        }

        // find classId
        auto classId = std::max_element(inFeatureMap + 5, inFeatureMap + BOX_FEATURE_MAP_LEN) - (inFeatureMap + 5);

        float classConf = inFeatureMap[5 + classId] * objectConf;
        if (classConf >= FINAL_CLASS_CONF_THRESHOLD) {
            float boxX = inFeatureMap[0];
            float boxY = inFeatureMap[1];
            float boxW = inFeatureMap[2];
            float boxH = inFeatureMap[3];

            DetectedObject obj{};
            // correct the position of the frame
            // first subtract the padding and then divide by the scaling factor
            obj.box.x = (boxX - boxW * 0.5f - xOffset) / ratio;
            obj.box.y = (boxY - boxH * 0.5f - yOffset) / ratio;
            obj.box.width = boxW / ratio;
            obj.box.height = boxH / ratio;
            obj.classId = (int) classId;
            obj.conf = classConf;
            outDetObjects.push_back(obj);
        }
    }
}

__global__ void parseYoloBoxGpuKernel(const float *inFeatureMap, DetectedObject *outDetObjects,
                                      int *atomicDetObjectsIdx, int xOffset, int yOffset, float ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= FEATURE_MAP_BOX_NUM) {
        return;
    }

    const float *ptr = inFeatureMap + idx * BOX_FEATURE_MAP_LEN;

    const float objectConf = ptr[4];

    if (objectConf < OBJECT_CONF_THRESHOLD) {
        return;
    }

    int classId;
    float classCondConf = 0;
    for (int i = 5; i < BOX_FEATURE_MAP_LEN; ++i) {
        if (ptr[i] > classCondConf) {
            classCondConf = ptr[i];
            classId = i - 5;
        }
    }


    float classConf = ptr[5 + classId] * objectConf;
    if (classConf >= FINAL_CLASS_CONF_THRESHOLD) {
        float boxX = ptr[0];
        float boxY = ptr[1];
        float boxW = ptr[2];
        float boxH = ptr[3];

        DetectedObject obj{};
        // 这里要修正框的位置，先减去补的padding，在除以缩放比例
        obj.box.x = (boxX - boxW * 0.5f - xOffset) / ratio;
        obj.box.y = (boxY - boxH * 0.5f - yOffset) / ratio;
        obj.box.width = boxW / ratio;
        obj.box.height = boxH / ratio;
        obj.classId = (int) classId;
        obj.conf = classConf;

        // get the atomic index
        const int atomicIdx = atomicAdd(&atomicDetObjectsIdx[0], 1);
        outDetObjects[atomicIdx] = obj;
    }
}

void yoloBoxGpu(const float *inFeatureMap, DetectedObject **outDevDetObjMem, int **outDevAtomicDetObjIdx,
                int xOffset, int yOffset, float ratio) {
    cudaMalloc(outDevDetObjMem, FEATURE_MAP_BOX_NUM * sizeof(DetectedObject));

    // record the index of the valid object of the atom saved in the array
    cudaMalloc(outDevAtomicDetObjIdx, sizeof(int));
    cudaMemset(*outDevAtomicDetObjIdx, 0, sizeof(int));


    const int blockSize = 1024;
    const int gridSize = (FEATURE_MAP_BOX_NUM + blockSize - 1) / blockSize; // grid大小
    // kernel function extracts the box data in the featureMap
    parseYoloBoxGpuKernel<<<gridSize, blockSize>>>(inFeatureMap, *outDevDetObjMem, *outDevAtomicDetObjIdx, xOffset,
                                                   yOffset, ratio);
}
