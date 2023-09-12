#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"

int main(int argc, char* argv[]) {
    std::string model_path = "../resource/weights/yolov5s_openvino_model/yolov5s.xml";
    std::string image_path = "../images/bus.jpg";
    det_input_t input_params;
    input_params.device = "CPU";
    input_params.threshold.score = 0.5;
    input_params.threshold.conf = 0.5;
    input_params.threshold.nms = 0.5;
    auto backend = Backend::GetBackend(BackendType::OPENVINO, input_params);

    backend->LoadModel(model_path);

    auto img = cv::imread(image_path);

    // 统计时间
    auto start_time = std::chrono::high_resolution_clock::now();
    auto batch_predictions = backend->Infer(img);

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Infer time: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << "ms"
              << std::endl;

    return 0;
}