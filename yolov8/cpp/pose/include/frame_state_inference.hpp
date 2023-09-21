#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include "math.h"
#include <algorithm>
#include <iomanip> // 包含格式化头文件

struct GridAndStride{
        int grid0;
        int grid1;
        int stride;
    };

struct Instance
{
    cv::Rect_<float> rect;
    std::vector<cv::Point> poses;
    std::vector<float> poses_prob;
    int label;
    float prob;
    
};
class FrameStateInference : public ExternalInterface
{
public:
    FrameStateInference();
    virtual ~FrameStateInference();
    virtual int detection(cv::Mat &inputImage, std::string &outputs);
    int loadModel(const std::string &modelPath, string configPath);

private:
    void *beforeProcess(std::vector<cv::Mat> &img);
    void afterProcess(void *data);
    void postProcess(std::vector<float *> outputs, int batch_size, std::vector<Json::Value> &res);

    void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
    void generateDetInstances(float *feat_blob, float prob_threshold, std::vector<Instance> &objects);
    void sortInstances(std::vector<Instance> &instances, int left, int right);
    void sortInstances(std::vector<Instance> &objects);
    void NMS(const std::vector<Instance> &instances, std::vector<int> &picked, float nms_threshold);
    std::vector<Instance> minNMS(std::vector<Instance> &instances, float nms_threshold);
    void postprocess(std::vector<Instance> instances,std::string &resultString);

private:
    int maxBatchSize_;
    int image_height_=640;
    int image_width_ = 640;
    int num_classes = 1;
    int num_poses = 17;
    float conf_thr_ = 0.25;
    int padding_value_=128;
    float nms_thr_ = 0.4;
    float min_thr_ = 0.6;
    int anchors_;
    TensorRT *tensorrt_;
};

#endif
