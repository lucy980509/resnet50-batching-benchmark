#include "batcher.hpp"

#include <algorithm>

#include "request_queue.hpp"

namespace{

using namespace std::chrono_literals;

const auto dynamic_wait = 20ms;
const auto adaptive_idle_wait = 5ms;

TimePoint batch_deadline(Policy policy, const Request& oldest,
                         const RequestQueue& queue, double estimated_inference_ms) {
    const auto now = Clock::now();
    if (policy == Policy::Dynamic){
        return now + dynamic_wait;
    }

    const auto wait = queue.empty() ? adaptive_idle_wait : dynamic_wait;
    const auto inference_budget = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double, std::milli>(estimated_inference_ms) + 2ms
    );
    const auto slo_deadline = oldest.arrived_at + slo - inference_budget;
    return std::min(now + wait, slo_deadline);
    }
}

Batch collect_batch(RequestQueue& queue, Policy policy, double estimated_inference_ms) {
    Batch batch;
    Request request;
    if (!queue.pop(request)) {
        return batch;
    }
    batch.requests[batch.size++] = request;

    if (policy == Policy::Static) {
        while (batch.size < max_batch_size && queue.pop(request)) {
            batch.requests[batch.size++] = request;
        }
        return batch;
    }

    const auto deadline = batch_deadline(
        policy, batch.requests[0], queue, estimated_inference_ms
    );

    while (batch.size < max_batch_size) {
        if (policy == Policy::Adaptive && queue.try_pop(request)) {
            batch.requests[batch.size++] = request;
            continue;
        }
        if (Clock::now() >= deadline) {
            break;
        }
        if (!queue.pop_until(request, deadline)) {
            break;
        }
        batch.requests[batch.size++] = request;
    }
    return batch;
}
