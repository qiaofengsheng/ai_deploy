#include <iostream>
#include <chrono>
#include "external_interface.hpp"

int main(int argc, char const *argv[])
{
    std::string onnxPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/crnn/model/crnn.trt";
    std::string configPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/crnn/model/config.ini";
    std::string dictPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/crnn/model/dicts.txt";
    ExternalInterface *fs = ExternalInterface::createModel(onnxPath, configPath,dictPath);
    char *pchImgPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/demo.png";
    cv::Mat matImg = cv::imread(pchImgPath);
    cv::Mat predScore;
    std::string res;
    fs->detection(matImg, res);
    std::cout<<res<<std::endl;
    return 0;
}
