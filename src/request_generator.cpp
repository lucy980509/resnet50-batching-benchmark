#include "request_generator.hpp"

#include <thread>

#include "request_queue.hpp"

using namespace std::chrono_literals;

const std::array<Workload, 4> workloads{{
    {"low", 40ms, 1},
    {"medium", 12ms, 1},
    {"high", 3ms, 1},
    {"bursty", 120ms, 8},
}};

void generate_requests(RequestQueue& queue, const Workload& workload,
                       std::size_t request_count, TimePoint start) {
    for (std::size_t i = 0; i < request_count; i++) {
        const auto slot = static_cast<int>(i / workload.burst_size);
        const auto arrival = start + workload.interval * slot;
        std::this_thread::sleep_until(arrival);
        const auto woke_at = Clock::now();
        Request request{arrival};
        request.woke_at = woke_at;
        request.image_index = i;
        if (!queue.push(request)) {
            break;
        }
    }
}
