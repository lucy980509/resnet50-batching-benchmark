#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include <onnxruntime_cxx_api.h>

class InferenceEngine {
public:
    InferenceEngine(const std::filesystem::path& model_path, std::size_t max_batch_size);

    void run(std::size_t batch_size);
    double warm_up();

private:
    std::size_t max_batch_size_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::vector<float> input_;
};
