#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {

double percentile(const std::vector<double>& sorted, double fraction) {
    const auto rank = static_cast<std::size_t>(
        std::ceil(sorted.size() * fraction)
    );
    return sorted[rank - 1];
}

}  // namespace

void write_csv_headers(std::ostream& summary, std::ostream& batches) {
    summary << std::fixed << std::setprecision(6)
            << "workload,policy,run,requests,batches,elapsed_s,throughput_req_s,"
               "p50_ms,p95_ms,p99_ms,within_100_ms,slo_percent,mean_batch_size,"
               "mean_queue_wait_ms,mean_batch_wait_ms\n";

    batches << std::fixed << std::setprecision(6)
            << "workload,policy,run,batch_number,batch_size,inference_ms\n";
}

void write_results(
    std::ostream& summary,
    std::ostream& batches,
    PolicyRun policy,
    const Workload& workload,
    int repetition,
    const RunResult& result
) {
    const auto& latencies = result.latencies_ms;
    const auto slo_ms = milliseconds(slo);

    const auto within_slo = std::upper_bound(
        latencies.begin(), latencies.end(), slo_ms
    ) - latencies.begin();

    const double slo_percent = 100.0 * within_slo / latencies.size();
    const double throughput = latencies.size() / result.elapsed_seconds;

    const double mean_batch_size =
        static_cast<double>(latencies.size()) / result.batches.size();

    const double mean_queue_wait_ms = std::accumulate(
        result.queue_waits_ms.begin(), result.queue_waits_ms.end(), 0.0
    ) / result.queue_waits_ms.size();

    const double mean_batch_wait_ms = std::accumulate(
        result.batch_waits_ms.begin(), result.batch_waits_ms.end(), 0.0
    ) / result.batch_waits_ms.size();

    const double p50 = percentile(latencies, 0.50);
    const double p95 = percentile(latencies, 0.95);
    const double p99 = percentile(latencies, 0.99);

    summary << workload.name << ',' << policy.name << ',' << repetition << ','
            << latencies.size() << ',' << result.batches.size() << ','
            << result.elapsed_seconds << ',' << throughput << ','
            << p50 << ',' << p95 << ',' << p99 << ','
            << within_slo << ',' << slo_percent << ',' << mean_batch_size << ','
            << mean_queue_wait_ms << ',' << mean_batch_wait_ms << '\n';

    for (std::size_t i = 0; i < result.batches.size(); ++i) {
        const auto& batch = result.batches[i];

        batches << workload.name << ',' << policy.name << ',' << repetition << ','
                << i + 1 << ',' << batch.size << ',' << batch.inference_ms << '\n';
    }

    std::cout << workload.name << '/' << policy.name << " run " << repetition
              << " | " << throughput << " req/s | P95 " << p95
              << " ms | SLO " << slo_percent << "%\n";
}

void write_request_results(
    std::ostream& requests,
    PolicyRun policy,
    const Workload& workload,
    int repetition,
    const RunResult& result
) {
    for (std::size_t i = 0; i < result.requests.size(); ++i) {
        const auto& request = result.requests[i];

        requests << workload.name << ','
                 << policy.name << ','
                 << repetition << ','
                 << i + 1 << ','
                 << request.arrival_delay_ms << ','
                 << request.queue_wait_ms << ','
                 << request.batch_wait_ms << ','
                 << request.inference_ms << ','
                 << request.latency_ms << ','
                 << request.wake_delay_ms << ','
                 << request.enqueue_after_wake_ms << '\n';
    }
}