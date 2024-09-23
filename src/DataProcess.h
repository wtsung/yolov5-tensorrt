#ifndef DATAPROCESS_H
#define DATAPROCESS_H

#include <vector>
#include "NvInferRuntime.h"

#include "BaseDef.h"

// when the number of input boxes is small, the CPU calculation speed is faster
void nmsCpu(std::vector<DetectedObject> &inDetBoxes, std::vector<DetectedObject> &outNMSBoxs, float nmsThreshold);

void nmsGpu(DetectedObject **inDevDetObjMem, int **inDevAtomicDetObjIdx, std::vector<DetectedObject> &outNmsObjs,
            float nmsThreshold);

void yoloBoxCpu(const float *inFeatureMap, std::vector<DetectedObject> &outDetObjects, int xOffset, int yOffset,
                float ratio);

void yoloBoxGpu(const float *inFeatureMap, DetectedObject **outDevDetObjMem, int **outDevAtomicDetObjIdx,
                int xOffset, int yOffset, float ratio);

#endif //DATAPROCESS_H
