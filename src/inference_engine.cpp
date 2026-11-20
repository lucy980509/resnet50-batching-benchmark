#include "inference_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include "request.hpp"

namespace {

const std::size_t image_elements = 3 * 224 * 224;

Ort::Session make_cuda_session(Ort::Env& env, const std::filesystem::path& model_path) {
    Ort::SessionOptions options;
    OrtCUDAProviderOptions cuda_options{};
    options.AppendExecutionProvider_CUDA(cuda_options);
    return Ort::Session(env, model_path.c_str(), options);
    }
}

InferenceEngine::InferenceEngine(const std::filesystem::path& model_path, std::size_t max_batch_size)
    : max_batch_size_(max_batch_size),
      env_(ORT_LOGGING_LEVEL_WARNING, "p3_inference"),
      session_(make_cuda_session(env_, model_path)),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_(max_batch_size * image_elements, 0.0f)
    {
        std::cout << "Inference engine ready.\n";
    }

void InferenceEngine::run(std::size_t batch_size) {
    const char* input_names[] = {"input"};
    const char* output_names[] = {"logits"};
    const std::int64_t shape[] = {
        static_cast<std::int64_t>(batch_size), 3, 224, 224
    };

    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_.data(), batch_size * image_elements, shape, 4
    );
    session_.Run(Ort::RunOptions{}, input_names, &tensor, 1, output_names, 1);
}

double InferenceEngine::warm_up() {
    double estimated_inference_ms = 0.0;
    for (std::size_t size = 1; size <= max_batch_size_; size++) {
        double steady_total_ms = 0.0;
        for (int pass = 0; pass < 3; pass++) {
            const auto start = Clock::now();
            run(size);
            const double elapsed_ms = milliseconds(Clock::now() - start);
            if (pass > 0) {
                steady_total_ms += elapsed_ms;
            }
        }
        estimated_inference_ms = std::max(estimated_inference_ms, steady_total_ms / 2.0);
    }
    return estimated_inference_ms;
}
