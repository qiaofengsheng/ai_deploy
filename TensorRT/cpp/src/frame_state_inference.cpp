#include "frame_state_inference.hpp"


FrameStateInference::FrameStateInference()
{
}

FrameStateInference::~FrameStateInference()
{
    delete tensorrt_;
}

void FrameStateInference::beforeProcess(cv::Mat &img,cv::Mat &img_data){
    cv::resize(img,img_data,cv::Size(600,600));
    img_data.convertTo(img_data, CV_32F);
}

void FrameStateInference::postProcess(std::vector<void *>outputs,Json::Value &outputs_){
    for(auto out:outputs){
        float sum_ = 0;
        int len = sizeof((float*)out)/sizeof(((float*)out)[0]);
        for(int i=0;i<len;++i){
            std::cout<<"i = "<<((float *)out)[i]<<std::endl;
            sum_+=exp(((float *)out)[i]);
        }
        int max_index=0;
        float max_confidence = exp(((float *)out)[0])/sum_;
        for(int i=1;i<len;++i){
            if(exp(((float *)out)[i]/sum_)>max_confidence){
                max_confidence=exp(((float *)out)[i])/sum_;
                max_index = i;
            }
        }
        outputs_["result_index"]=max_index;
        outputs_["confidence"] = max_confidence;
    }
    
}

int FrameStateInference::loadModel(const std::string &modelPath,const int &batchSize)
{
    tensorrt_ = new TensorRT();
    int status = tensorrt_->loadModel(modelPath, batchSize);
    return status;
}

int FrameStateInference::detection(cv::Mat &input_imgs, Json::Value &outputs)
{
    cv::Mat img_data;
    beforeProcess(input_imgs,img_data);    
    const int batch_size=1;
    std::vector<void *> output_;
    int status = tensorrt_->doInference((float*)img_data.data,batch_size,output_);
    postProcess(output_,outputs);
    return status;
}
