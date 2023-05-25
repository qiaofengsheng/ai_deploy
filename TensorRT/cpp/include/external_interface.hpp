#ifndef __EXTERNAL_INTERFACE_H__
#define __EXTERNAL_INTERFACE_H__
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "json.h"
class ExternalInterface
{

public:
    ExternalInterface();
    virtual ~ExternalInterface();

    static ExternalInterface *createModel(const std::string &modelPath, std::string configPath);

    virtual int detection(std::vector<cv::Mat> &input_imgs, std::vector<Json::Value> &outputs) = 0;
};

#endif