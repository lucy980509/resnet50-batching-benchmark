#pragma once

#include <chrono>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Request {
    TimePoint arrived_at{};
    TimePoint enqueued_at{};
    TimePoint dequeued_at{};
    TimePoint woke_at{};
};

inline double milliseconds(Clock::duration elapsed) {
    return std::chrono::duration<double, std::milli>(elapsed).count();
}
