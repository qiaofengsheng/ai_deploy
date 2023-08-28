#include <iostream>
#include <chrono>
#include "external_interface.hpp"

int main(int argc, char const *argv[])
{
    std::string onnxPath = "/home/qfs/project/server_project/teach_common_ocr_yolox/ai_deploy/model/model_final.trt";
    // const int batchSize = 1;
    // cv::Mat img = cv::imread("/data/qfs/project/study/pytorch-UNet/data/JPEGImages/000799.png");
    // ExternalInterface *detector = ExternalInterface::createModel(onnxPath);
    // TensorRT *tensorrt_ = new TensorRT();
    // tensorrt_->loadModel(onnxPath, {"input"}, {"output"}, {{1, 3, 600, 600}}, {{1, 2}}, 4);
    std::string configPath = "/home/qfs/project/server_project/teach_common_ocr_yolox/ai_deploy/model/config.ini";
    ExternalInterface *fs = ExternalInterface::createModel(onnxPath, configPath);

    for(int i=0;i<50;i++){
        char *pchImgPath = "/home/qfs/project/server_project/teach_common_ocr_yolox/ai_deploy/2y8sig0vag.png";
        cv::Mat matImg = cv::imread(pchImgPath);
        std::string outputs;
        auto start = std::chrono::high_resolution_clock::now();

        fs->detection(matImg, outputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long long milliseconds = duration.count();

        std::cout << "运行时间：" << milliseconds << " 毫秒" << std::endl;

        std::cout<<outputs<<std::endl;
    }
    
    // for (auto output : outputs)
    // {
    //     std::cout << output << std::endl;
    // }
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cv::Mat matImg = cv::imread(pchImgPath);
    //     std::cout << matImg.rows << std::endl;
    //     cv::Mat matRzImg;
    //     cv::resize(matImg, matRzImg, cv::Size(600, 600));
    //     cv::Mat matF32Img;
    //     matRzImg.convertTo(matF32Img, CV_32FC1);
    //     matF32Img = matF32Img / 255.;
    //     std::vector<float> inputData(3 * 600 * 600);
    //     float *dataPtr = inputData.data();

    //     for (int c = 0; c < 3; ++c)
    //     {
    //         for (int h = 0; h < 600; ++h)
    //         {
    //             for (int w = 0; w < 600; ++w)
    //             {
    //                 dataPtr[c * 600 * 600 + h * 600 + w] = matF32Img.at<cv::Vec3f>(h, w)[c];
    //             }
    //         }
    //     }
    //     std::vector<void *> output_;
    //     float *input_ = new float[3 * 600 * 600 * 2];
    //     memcpy(input_, dataPtr, 3 * 600 * 600 * sizeof(float));
    //     memcpy(input_ + 3 * 600 * 600, dataPtr, 3 * 600 * 600 * sizeof(float));
    //     tensorrt_->doBatchInference({input_}, 2, output_);
    // }

    // // Json::Value outputs;
    // // detector->detection(img,outputs);
    // // std::cout<<outputs<<std::endl;

    return 0;
}
