#ifndef __BACKEND__B_OPENVINO_HPP__
#define __BACKEND__B_OPENVINO_HPP__

#include <memory>
#include <opencv2/opencv.hpp>

#include "backend/interface.hpp"
#include "openvino/openvino.hpp"
#include "utils/extend_print.hpp"

class OpenVINOBackend : public BaseDetectionBackend {
private:
    ov::CompiledModel compiled_model;
    // ov::InferRequest infer_request;// FIXME: is a member variable necessary?
public:
    OpenVINOBackend(const det_input_t& input_params);
    ~OpenVINOBackend();
    void LoadModel(const std::string& model_path);
    det_output_t Infer(const cv::Mat& img);
    std::vector<std::string> GetDevice() const;

    OpenVINOBackend(const OpenVINOBackend&) = delete;
    OpenVINOBackend(OpenVINOBackend&&) = delete;
    OpenVINOBackend& operator=(const OpenVINOBackend&) = delete;
    OpenVINOBackend& operator=(OpenVINOBackend&&) = delete;
};

#endif  // __BACKEND__B_OPENVINO_HPP__
