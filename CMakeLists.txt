cmake_minimum_required(VERSION 3.20)
project(AdaptiveBatchingInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_DIR "C:/onnxruntime/onnxruntime-win-x64-gpu-1.26.0"
    CACHE PATH "ONNX Runtime GPU installation")
set(CUDA_BIN_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
    CACHE PATH "CUDA runtime DLL directory")
set(CUDNN_BIN_DIR "C:/Program Files/NVIDIA/CUDNN/v9.8/bin/12.8"
    CACHE PATH "cuDNN runtime DLL directory")

find_package(Threads REQUIRED)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(p3_inference
    src/main.cpp
    src/benchmark.cpp
    src/request_generator.cpp
    src/batcher.cpp
    src/inference_engine.cpp
    src/metrics.cpp
)

target_include_directories(p3_inference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
)
target_link_libraries(p3_inference PRIVATE Threads::Threads "${ONNXRUNTIME_LIBRARY}")

if(MSVC)
    target_compile_options(p3_inference PRIVATE /utf-8)
endif()

if(WIN32)
    file(GLOB RUNTIME_DLLS CONFIGURE_DEPENDS
        "${ONNXRUNTIME_DIR}/lib/*.dll"
        "${CUDA_BIN_DIR}/*.dll"
        "${CUDNN_BIN_DIR}/*.dll"
    )
    foreach(DLL_FILE IN LISTS RUNTIME_DLLS)
        add_custom_command(TARGET p3_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL_FILE}" "$<TARGET_FILE_DIR:p3_inference>"
            VERBATIM
        )
    endforeach()
endif()

add_custom_command(TARGET p3_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/models" "$<TARGET_FILE_DIR:p3_inference>/models"
    VERBATIM
)
