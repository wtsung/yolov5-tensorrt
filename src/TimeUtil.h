#ifndef TIMEUTIL_H
#define TIMEUTIL_H

#include <chrono>

class TimeUtil {
public:
    TimeUtil() {
        // start timing in construction
        restart();
    }

    // restart the timer
    void restart() {
        mStartTime = std::chrono::high_resolution_clock::now();
    }

    // stop and return the duration (milliseconds)
    double duration() {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - mStartTime;
        return duration.count() * 1000;
    }

private:
    std::chrono::high_resolution_clock::time_point mStartTime;
};

#endif //TIMEUTIL_H
