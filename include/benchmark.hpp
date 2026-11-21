#pragma once

#include "metrics.hpp"

class InferenceEngine;

RunResult benchmark(
    InferenceEngine& engine,
    Policy policy,
    const Workload& workload,
    std::size_t request_count,
    double estimated_inference_ms,
    const std::vector<std::vector<float>>& images
);