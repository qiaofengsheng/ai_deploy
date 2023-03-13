#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include"math.h"

class FrameStateInference : public ExternalInterface {
    public:
        FrameStateInference();
        virtual ~FrameStateInference();
        virtual int detection(cv::Mat& input_imgs,Json::Value &outputs);
        int loadModel(const std::string &modelPath,const int &batchSize);

    protected:
        void beforeProcess(cv::Mat &img,cv::Mat &img_data);
        void postProcess(std::vector<void *>outputs,Json::Value &outputs_);
    private:
        TensorRT* tensorrt_;
};

#endif
