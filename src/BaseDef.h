#ifndef BASEDEF_H
#define BASEDEF_H

#include <string>
#include <unordered_map>

#define FEATURE_MAP_BOX_NUM 25200
#define BOX_FEATURE_MAP_LEN 85
#define OBJECT_CONF_THRESHOLD 0.01
#define FINAL_CLASS_CONF_THRESHOLD 0.25

struct ObjectBox {
    float x;
    float y;
    float width;
    float height;
};

struct DetectedObject {
    ObjectBox box;
    int classId;
    float conf;
};

extern std::unordered_map<int, std::string> classIdToNameMap;

inline std::string getClassName(int classId) {
    return classIdToNameMap.find(classId) != classIdToNameMap.end() ? classIdToNameMap[classId] : "unknown";
}

#endif //BASEDEF_H
