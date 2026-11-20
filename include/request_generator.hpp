#pragma once

#include <array>
#include <chrono>
#include <cstddef>

#include "request.hpp"

class RequestQueue;

struct Workload {
    const char* name;
    std::chrono::milliseconds interval;
    std::size_t burst_size;
};

extern const std::array<Workload, 4> workloads;

void generate_requests(RequestQueue& queue, const Workload& workload,
                       std::size_t request_count, TimePoint start);
