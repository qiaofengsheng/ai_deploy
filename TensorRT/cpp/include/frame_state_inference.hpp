#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include "math.h"

class FrameStateInference : public ExternalInterface
{
public:
    FrameStateInference();
    virtual ~FrameStateInference();
    virtual int detection(std::vector<cv::Mat> &input_imgs, std::vector<Json::Value> &outputs);
    int loadModel(const std::string &modelPath, string configPath);

protected:
    void *beforeProcess(std::vector<cv::Mat> &img);
    void postProcess(std::vector<float *> outputs, int batch_size, std::vector<Json::Value> &res);

private:
    int maxBatchSize_;
    TensorRT *tensorrt_;
};

#endif
