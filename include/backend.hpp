#ifndef __BACKEND_HPP__
#define __BACKEND_HPP__
#include <memory>

#include "backend/b_openvino.hpp"
#include "backend/interface.hpp"

enum class BackendType {
    ONNX,
    OPENVINO,
    TensorRT,
};

class Backend {
public:
    Backend() = delete;
    ~Backend() = delete;
    static std::unique_ptr<BaseDetectionBackend> GetBackend(BackendType backend_type, const det_input_t& input_params) {
        switch (backend_type) {
        case BackendType::OPENVINO:
            return std::make_unique<OpenVINOBackend>(input_params);
            break;
        default:
            return nullptr;
            break;
        }
    }
};

#endif  // __BACKEND_HPP__