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

    virtual int detection(cv::Mat &inputImage,cv::Mat &predScore,cv::Mat &predMat) = 0;
};

#endif