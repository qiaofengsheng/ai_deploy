#include "external_interface.hpp"

int main(int argc, char const *argv[])
{
    std::string onnxPath = "/data/qfs/project/study/ai_deploy/TensorRT/cpp/model/image_cls_resnet18_600_dym.trt";
    // const int batchSize = 1;
    // cv::Mat img = cv::imread("/data/qfs/project/study/pytorch-UNet/data/JPEGImages/000799.png");
    // ExternalInterface *detector = ExternalInterface::createModel(onnxPath);
    // TensorRT *tensorrt_ = new TensorRT();
    // tensorrt_->loadModel(onnxPath, {"input"}, {"output"}, {{1, 3, 600, 600}}, {{1, 2}}, 4);
    std::string configPath = "/data/qfs/project/study/ai_deploy/TensorRT/cpp/model/config.ini";
    ExternalInterface *fs = ExternalInterface::createModel(onnxPath, configPath);

    char *pchImgPath = "/data/qfs/project/server_project/xuepaipai_teach_common_ocr/git_code/efficientnet_cls_batch_cpp/data/images/testing/0a23a93b2ee11eafe6409e4441ce9b91.jpeg";
    cv::Mat matImg = cv::imread(pchImgPath);
    std::vector<Json::Value> outputs;
    std::vector<cv::Mat> input_imgs = {matImg, matImg};
    fs->detection(input_imgs, outputs);
    for (auto output : outputs)
    {
        std::cout << output << std::endl;
    }
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
