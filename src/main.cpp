#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "benchmark.hpp"
#include "inference_engine.hpp"
#include "metrics.hpp"
#include "dataset.hpp"

int main() {
    try {
        const std::filesystem::path model_path =
            "models/resnet50/resnet50_v2_fp32.onnx";
        const int request_count = 1000;
        const int repetitions = 3;

        const auto images = read_prepared_images(
            kPreparedImagePath,
            kPreparedImageCount,
            3 * 224 * 224,
            0,
            request_count
        );

        std::cout << "Loaded images: " << images.size() << '\n';

        InferenceEngine engine(model_path, max_batch_size);

        std::ofstream summary("benchmark_summary.csv");
        std::ofstream batches("benchmark_batches.csv");
        std::ofstream requests("benchmark_requests.csv");

        summary.exceptions(std::ios::failbit | std::ios::badbit);
        batches.exceptions(std::ios::failbit | std::ios::badbit);
        requests.exceptions(std::ios::failbit | std::ios::badbit);

        write_csv_headers(summary, batches);

        requests << std::fixed << std::setprecision(6);
        requests << "workload,policy,run,request_number,"
                    "arrival_delay_ms,queue_wait_ms,batch_wait_ms,"
                    "inference_ms,latency_ms,wake_delay_ms,"
                    "enqueue_after_wake_ms\n";

        std::cout << std::fixed << std::setprecision(2);

        for (const auto& workload : workloads) {
            for (int repetition = 1; repetition <= repetitions; repetition++) {
                for (std::size_t i = 0; i < policies.size(); i++) {
                    const auto policy =
                        policies[(i + repetition - 1) % policies.size()];

                    const double estimate = engine.warm_up();

                    const auto result = benchmark(
                        engine,
                        policy.policy,
                        workload,
                        request_count,
                        estimate,
                        images
                    );

                    write_results(
                        summary, batches, policy, workload, repetition, result
                    );

                    write_request_results(
                        requests, policy, workload, repetition, result
                    );
                }
            }
        }

        summary.close();
        batches.close();
        requests.close();

        std::cout << "Saved benchmark_summary.csv, benchmark_batches.csv, "
                     "and benchmark_requests.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}