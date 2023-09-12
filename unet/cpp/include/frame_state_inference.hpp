#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include "math.h"
#include <random>
struct GridAndStride{
        int grid0;
        int grid1;
        int stride;
    };

struct Instance
{
    cv::Rect_<float> rect;
    std::vector<cv::Point> polys;
    int label;
    float prob;
};
class FrameStateInference : public ExternalInterface
{
public:
    FrameStateInference();
    virtual ~FrameStateInference();
    virtual int detection(cv::Mat &inputImage,cv::Mat &predScore,cv::Mat &predMat);
    int loadModel(const std::string &modelPath, string configPath);

private:
    void *beforeProcess(std::vector<cv::Mat> &img);
    void afterProcess(void *data);
    // void postProcess(std::vector<float *> outputs, int batch_size,cv::Mat &outMat,cv::Mat &outScore);
    std::vector<cv::Mat> postProcess(std::vector<float *> outputs, int batch_size);

private:
    int maxBatchSize_;
    int image_height_=512;
    int image_width_ = 512;
    int output_height = 512;
    int output_width = 512;
    int num_classes = 21;
    int padding_value_=128;
    std::vector<cv::Vec3b> colors;
    TensorRT *tensorrt_;
};

#endif
