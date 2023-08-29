#ifndef __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FRAME_STATE_DETECTOR_HPP__

#include <map>
#include <string>
#include "external_interface.hpp"
#include "tensorrt.hpp"
#include "json.h"
#include "math.h"

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
    virtual int detection(cv::Mat &inputImage, std::string &outputs);
    int loadModel(const std::string &modelPath, string configPath);

private:
    void *beforeProcess(std::vector<cv::Mat> &img);
    void afterProcess(void *data);
    void postProcess(std::vector<float *> outputs, int batch_size, std::vector<Json::Value> &res);

    void generate_grids_and_stride(std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
    void generateDetInstances(std::vector<GridAndStride> grid_strides, float *feat_blob, float prob_threshold, std::vector<Instance> &objects);
    void sortInstances(std::vector<Instance> &instances, int left, int right);
    void sortInstances(std::vector<Instance> &objects);
    void obbToPolys(float x_center, float y_center, float w, float h, float theta, std::vector<cv::Point> &polys);
    void NMS(const std::vector<Instance> &instances, std::vector<int> &picked, float nms_threshold);
    std::vector<Instance> minNMS(std::vector<Instance> &instances, float nms_threshold);
    void postprocess(std::vector<Instance> instances,std::string &resultString);

private:
    int maxBatchSize_;
    int image_height_=800;
    int image_width_ = 800;
    int num_classes = 19;
    float conf_thr_ = 0.01;
    int padding_value_=255;
    float nms_thr_ = 0.5;
    float min_thr_ = 0.6;
    int anchors_;
    TensorRT *tensorrt_;
};

#endif
