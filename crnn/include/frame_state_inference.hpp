#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include "math.h"
#include <random>
class FrameStateInference : public ExternalInterface
{
public:
    FrameStateInference();
    virtual ~FrameStateInference();
    virtual int detection(cv::Mat &inputImage,std::string &output);
    int loadModel(const std::string &modelPath, string configPath,std::string dictPath);

private:
    void *beforeProcess(std::vector<cv::Mat> &img);
    void afterProcess(void *data);
    int initDict(std::string dictPath);
    // void postProcess(std::vector<float *> outputs, int batch_size,cv::Mat &outMat,cv::Mat &outScore);
    std::string postProcess(std::vector<float *> outputs, int batch_size);

private:
    int maxBatchSize_;
    int image_height_=32;
    int image_width_ = 100;
    int seq_len = 26;
    int num_classes = 37;
    int padding_value_=128;
    std::vector<std::string> word_dicts;
    TensorRT *tensorrt_;
};

#endif
