#include <iostream>
#include <chrono>
#include "external_interface.hpp"

int main(int argc, char const *argv[])
{
    std::string onnxPath = "./model/models.trt";
    std::string configPath = "./model/config.ini";
    ExternalInterface *fs = ExternalInterface::createModel(onnxPath, configPath);
    char *pchImgPath = "./street.jpg";
    cv::Mat matImg = cv::imread(pchImgPath);
    cv::Mat predScore;
    cv::Mat resultMat(matImg.rows,matImg.cols,CV_8UC3,cv::Scalar(0,0,0));
    fs->detection(matImg, predScore,resultMat);
    cv::imwrite("./res.jpg",resultMat);
    return 0;
}
