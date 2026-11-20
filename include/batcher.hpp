#pragma once

#include <array>
#include <chrono>
#include <cstddef>

#include "request.hpp"

class RequestQueue;

inline const std::size_t max_batch_size = 16;
inline const auto slo = std::chrono::milliseconds(100);

enum class Policy { Static, Dynamic, Adaptive };

struct PolicyRun {
    Policy policy;
    const char* name;
};

inline const std::array<PolicyRun, 3> policies{{
    {Policy::Static, "static"},
    {Policy::Dynamic, "dynamic"},
    {Policy::Adaptive, "adaptive"},
}};

struct Batch {
    std::array<Request, max_batch_size> requests;
    std::size_t size = 0;
};

Batch collect_batch(RequestQueue& queue, Policy policy, double estimated_inference_ms);
