#ifndef __BACKEND_HPP__
#define __BACKEND_HPP__
#include <memory>

#ifdef WITH_OPENVINO
#include "backend/b_openvino.hpp"
#endif
#include "backend/interface.hpp"

enum class BackendType {
  ONNX,
#ifdef WITH_OPENVINO
  OPENVINO,
#endif
  TENSORRT,
};

class Backend {
public:
  Backend() = delete;
  ~Backend() = delete;
  static std::unique_ptr<BaseDetectionBackend>
  GetBackend(BackendType backend_type, const det_input_t &input_params) {
    switch (backend_type) {
#ifdef WITH_OPENVINO
    case BackendType::OPENVINO:
      return std::make_unique<OpenVINOBackend>(input_params);
      break;
#endif

    default:
      return nullptr;
      break;
    }
  }
};

#endif // __BACKEND_HPP__