#include <iostream>
#include <chrono>
#include "external_interface.hpp"
#include <ctime>
#include <random>
int main(int argc, char const *argv[])
{
    std::vector<cv::Scalar> colors;
    std::default_random_engine e;
    std::uniform_int_distribution<int> u(0,255);
    e.seed(time(0));
    for(int c=0;c<80;++c){
        colors.push_back(cv::Scalar(u(e),u(e),u(e)));
    }
    std::string onnxPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/yolov8/cpp/seg/model/yolov8n-seg.trt";
    std::string configPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/yolov8/cpp/seg/model/config.ini";
    ExternalInterface *fs = ExternalInterface::createModel(onnxPath, configPath);

    // for(int i=0;i<50;i++){
    char *pchImgPath = "/home/qfs/project/server_project/study/tensorrt/ai_deploy/street.jpg";
    cv::Mat matImg = cv::imread(pchImgPath);
    std::vector<outStruct> outputs;
    // auto start = std::chrono::high_resolution_clock::now();

    fs->detection(matImg, outputs);
    // int c=0;
    for(auto out:outputs){
        cv::Point left_top(out.x1,out.y1);
        cv::Point right_bottom(out.x2,out.y2);
        cv::rectangle(matImg,left_top,right_bottom,colors[out.cls],2);
        cv::Mat color_mask(matImg.size(), CV_8UC3, colors[out.cls]);
        // 将彩色蒙版应用到原始图像上
        cv::Mat result;
        matImg.copyTo(result, out.mask);
        cv::Mat red_mask(matImg.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        red_mask.setTo(colors[out.cls], out.mask);
        cv::addWeighted(matImg, 1, red_mask, 0.5, 0, matImg);
        // 将彩色蒙版叠加到结果图像上
        // cv::imwrite("/home/qfs/project/server_project/study/tensorrt/ai_deploy/yolov8/cpp/seg/"+std::to_string(c)+".jpg",red_mask);
        // cv::add(result, color_mask, result, out.mask);
        // c+=1;
    }
    cv::imwrite("/home/qfs/project/server_project/study/tensorrt/ai_deploy/street_c.jpg",matImg);


    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // long long milliseconds = duration.count();
    // std::cout << "运行时间：" << milliseconds << " 毫秒" << std::endl;
    // Json::Value root;
    // Json::Reader reader;
    // if(reader.parse(outputs, root)){
    //     for(auto data:root["data"]){
    //         cv::Point left_top(data["quadrangle"]["x1"].asInt(),data["quadrangle"]["y1"].asInt());
    //         cv::Point right_bottom(data["quadrangle"]["x2"].asInt(),data["quadrangle"]["y2"].asInt());
    //         int cls_index = data["class"].asInt();
    //         float conf = data["confidence"].asFloat();
    //         cv::rectangle(matImg,left_top,right_bottom,colors[cls_index],2);
    //         cv::putText(matImg,std::to_string(cls_index)+" "+std::to_string(conf),left_top,cv::FONT_HERSHEY_COMPLEX,0.5,colors[cls_index]);
    //     }
    //     cv::imwrite("/home/qfs/project/server_project/study/tensorrt/ai_deploy/street_c.jpg",matImg);
    // } 



    return 0;
}
