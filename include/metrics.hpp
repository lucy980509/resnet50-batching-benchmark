#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

#include "batcher.hpp"
#include "request_generator.hpp"

struct BatchResult {
    std::size_t size;
    double inference_ms;
};

struct RequestResult {
    double arrival_delay_ms;
    double queue_wait_ms;
    double batch_wait_ms;
    double inference_ms;
    double latency_ms;
    double wake_delay_ms;
    double enqueue_after_wake_ms;
};

struct RunResult {
    std::vector<double> latencies_ms;
    std::vector<double> queue_waits_ms;
    std::vector<double> batch_waits_ms;
    std::vector<BatchResult> batches;
    std::vector<RequestResult> requests;
    double elapsed_seconds = 0.0;
};

void write_csv_headers(std::ostream& summary, std::ostream& batches);

void write_results(
    std::ostream& summary,
    std::ostream& batches,
    PolicyRun policy,
    const Workload& workload,
    int repetition,
    const RunResult& result
);

void write_request_results(
    std::ostream& requests,
    PolicyRun policy,
    const Workload& workload,
    int repetition,
    const RunResult& result
);