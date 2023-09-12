#include "backend/b_openvino.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "backend/interface.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/extend_print.hpp"

OpenVINOBackend::OpenVINOBackend(const det_input_t& input_params) : BaseDetectionBackend(input_params) {
    auto devices = this->GetDevice();
    std::string devices_str;
    for (auto&& device : devices) {
        devices_str += device + ", ";
    }
    OPENVINO_ASSERT(std::find(devices.begin(), devices.end(), this->input_params.device) != devices.end(),
                    "Device name is not in available devices: " + devices_str);
}

OpenVINOBackend::~OpenVINOBackend() {}

void OpenVINOBackend::LoadModel(const std::string& model_path) {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
    OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");
    // -------- Step 3. Set up input

    ov::element::Type input_type = ov::element::u8;
    const ov::Layout tensor_layout{"NHWC"};

    // -------- Step 4. Configure preprocessing --------

    ov::preprocess::PrePostProcessor ppp(model);

    // 1) Set input tensor information:
    // - input() provides information about a single model input
    // - reuse precision and shape from already available `input_tensor`
    // - layout of data is 'NHWC'
    ppp.input().tensor().set_element_type(input_type).set_layout(tensor_layout);
    // 2) Adding explicit preprocessing steps:
    // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    // - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    // 4) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout("NCHW");
    // 5) Set output tensor information:
    // - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 6) Apply preprocessing modifying the original 'model'
    model = ppp.build();

    // -------- Step 5. Loading a model to the device --------
    this->compiled_model = core.compile_model(model, this->input_params.device);
    if (this->compiled_model.input().get_shape().size() == 4) {
        auto input_shape = this->compiled_model.input().get_shape();
        this->input_params.batch_size = input_shape[0];
        this->input_params.height = input_shape[1];
        this->input_params.width = input_shape[2];
        this->input_params.channel = input_shape[3];
    } else
        throw std::runtime_error("Model input shape is not BCHW");
    if (this->compiled_model.output().get_shape().size() != 3)
        throw std::runtime_error("Model output shape is not BNS");
    auto output_shape = this->compiled_model.output().get_shape();
    // this->batch_predictions = det_output_t(output_shape[0], std::vector<detetcion_output_t>(output_shape[1]));
}

det_output_t OpenVINOBackend::Infer(const cv::Mat& _img) {
    cv::Mat img;
    cv::resize(_img,
               img,
               cv::Size(this->compiled_model.input().get_shape()[1], this->compiled_model.input().get_shape()[2]));

    auto infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(
        ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), img.data));
    infer_request.infer();

    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();

    float* detections = output_tensor.data<float>();
    det_output_t batch_predictions;
    for (int b = 0; b < output_shape[0]; b++) {
        std::vector<detetcion_output_t> predictions;
        std::vector<cv::Rect> boxes;
        for (int n = 0; n < output_shape[1]; n++) {
            float* detection = &detections[(b * output_shape[1] + n) * output_shape[2]];

            if (detection[4] < this->input_params.threshold.conf)
                continue;

            int max_class_id = 5;
            for (int i = 5; i < output_shape[2]; i++) {
                if (detection[i] > detection[max_class_id])
                    max_class_id = i;
            }
            if (detection[max_class_id] < this->input_params.threshold.score)
                continue;

            auto box = cv::Rect(cv::Point(detection[0], detection[1]), cv::Point(detection[2], detection[3]));
            detetcion_output_t prediction{
                box,
                detection[4],
                max_class_id - 5,
                detection[max_class_id],
            };
            boxes.push_back(box);

            predictions.emplace_back(prediction);
        }

        batch_predictions.emplace_back(predictions);
    }

    return batch_predictions;  // TODO: enable RVO
}

std::vector<std::string> OpenVINOBackend::GetDevice() const {
    ov::Core core;
    auto availableDevices = core.get_available_devices();
#ifdef DEBUG
    {
        for (auto&& device : availableDevices) {
            std::cout << device << std::endl;
        }
    }
#endif
    return availableDevices;
}