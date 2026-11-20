#include "benchmark.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <thread>

#include "inference_engine.hpp"
#include "request_queue.hpp"

using namespace std::chrono_literals;

RunResult benchmark(InferenceEngine& engine, Policy policy, const Workload& workload,
                    std::size_t request_count, double estimated_inference_ms) {
    RequestQueue queue;
    RunResult result;
    result.latencies_ms.reserve(request_count);
    result.queue_waits_ms.reserve(request_count);
    result.batch_waits_ms.reserve(request_count);
    result.requests.reserve(request_count);
    result.batches.reserve(request_count);

    std::exception_ptr producer_error;
    const auto start = Clock::now() + 20ms;
    std::thread producer([&] {
        try {
            generate_requests(queue, workload, request_count, start);
        } catch (...) {
            producer_error = std::current_exception();
        }
        queue.close();
    });

    auto last_finished = start;
    try {
        while (true) {
            const Batch batch = collect_batch(queue, policy, estimated_inference_ms);
            if (batch.size == 0) {
                break;
            }

            const auto inference_start = Clock::now();
            engine.run(batch.size);
            last_finished = Clock::now();
            result.batches.push_back({batch.size, milliseconds(last_finished - inference_start)});

            for (std::size_t i = 0; i < batch.size; ++i) {
                const auto& request = batch.requests[i];

                const double arrival_delay_ms = milliseconds(request.enqueued_at - request.arrived_at);
                const double queue_wait_ms = milliseconds(request.dequeued_at - request.enqueued_at);
                const double batch_wait_ms = milliseconds(inference_start - request.dequeued_at);
                const double inference_ms = milliseconds(last_finished - inference_start);
                const double latency_ms = milliseconds(last_finished - request.arrived_at);
                const double wake_delay_ms = milliseconds(request.woke_at - request.arrived_at);
                const double enqueue_after_wake_ms = milliseconds(request.enqueued_at - request.woke_at);
                result.latencies_ms.push_back(latency_ms);
                result.queue_waits_ms.push_back(queue_wait_ms);
                result.batch_waits_ms.push_back(batch_wait_ms);
    
                result.requests.push_back({
                    arrival_delay_ms,
                    queue_wait_ms,
                    batch_wait_ms,
                    inference_ms,
                    latency_ms,
                    wake_delay_ms,
                    enqueue_after_wake_ms
                });
            }
        }
    } catch (...) {
        queue.close();
        producer.join();
        throw;
    }

    producer.join();
    if (producer_error) {
        std::rethrow_exception(producer_error);
    }
    if (result.latencies_ms.size() != request_count) {
        throw std::runtime_error("Benchmark did not process every request.");
    }

    result.elapsed_seconds = std::chrono::duration<double>(last_finished - start).count();
    std::sort(result.latencies_ms.begin(), result.latencies_ms.end());
    return result;
}
