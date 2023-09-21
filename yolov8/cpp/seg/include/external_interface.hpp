#ifndef __EXTERNAL_INTERFACE_H__
#define __EXTERNAL_INTERFACE_H__
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "json.h"

struct outStruct{
    int x1;
    int y1;
    int x2;
    int y2;
    int cls;
    float conf;
    cv::Mat mask;
};

class ExternalInterface
{

public:
    ExternalInterface();
    virtual ~ExternalInterface();

    static ExternalInterface *createModel(const std::string &modelPath, std::string configPath);

    virtual int detection(cv::Mat &inputImage, std::vector<outStruct> &outputs) = 0;
};

#endif