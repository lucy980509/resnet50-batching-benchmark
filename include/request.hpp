#pragma once

#include <chrono>
#include <cstddef>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Request {
    TimePoint arrived_at{};
    TimePoint enqueued_at{};
    TimePoint dequeued_at{};
    TimePoint woke_at{};
    std::size_t image_index{};
};

inline double milliseconds(Clock::duration elapsed) {
    return std::chrono::duration<double, std::milli>(elapsed).count();
}
