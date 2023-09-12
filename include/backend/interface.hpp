#ifndef __BACKEND__INTERFACE_HPP__
#define __BACKEND__INTERFACE_HPP__

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

typedef struct {
    std::string device;
    unsigned int batch_size;
    unsigned int height;
    unsigned int width;
    unsigned int channel;
    struct {
        float conf;
        float score;
        float nms;
    } threshold;
} detection_input_t;

typedef struct {
    cv::Rect boxes;
    float obj_conf;
    int class_id;
    float class_conf;
} detetcion_output_t;

template <typename _TInput, typename _TOutput>
class IBackend {
protected:
    _TInput input_params;

public:
    explicit IBackend(const _TInput& input_params) : input_params(input_params){};
    virtual ~IBackend(){};
    virtual void LoadModel(const std::string& model_path) = 0;
    virtual _TOutput Infer(const cv::Mat& img) = 0;
    virtual std::vector<std::string> GetDevice() const = 0;
};

using det_input_t = detection_input_t;
using det_output_t = std::vector<std::vector<detetcion_output_t>>;
class BaseDetectionBackend : public IBackend<det_input_t, det_output_t> {
public:
    explicit BaseDetectionBackend(const det_input_t& input_params) : IBackend(input_params){};
};

#endif  // __BACKEND_INTERFACE_HPP__
