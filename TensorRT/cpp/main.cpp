#include"tensorrt.hpp"
#include"external_interface.hpp"
#include"frame_state_inference.hpp"

int main(int argc, char const *argv[])
{
    std::string onnxPath = "/data/qfs/project/study/qfs_trt/image_cls_resnet18_0.975_600.trt";
    const int batchSize=1;
    cv::Mat img = cv::imread("/data/qfs/project/study/qfs_trt/z.jpg");
    ExternalInterface* detector = ExternalInterface::createModel(onnxPath,batchSize);
    Json::Value outputs;
    detector->detection(img,outputs);
    std::cout<<outputs<<std::endl;

    return 0;
}
