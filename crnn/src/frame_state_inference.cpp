#include "frame_state_inference.hpp"

FrameStateInference::FrameStateInference()
{

}

FrameStateInference::~FrameStateInference()
{
    delete tensorrt_;
}

void FrameStateInference::afterProcess(void *data)
{
    float *inputData = static_cast<float *>(data);
    delete[] inputData; // 释放分配的内存块
}

void *FrameStateInference::beforeProcess(std::vector<cv::Mat> &img)
{

    float *inputData = new float[img.size() * 1 * this->image_height_ * this->image_width_];

    for (int i = 0; i < img.size(); ++i)
    {
        // cv::cvtColor(img[i],img[i],cv::COLOR_BGR2RGB);
        cv::Mat img_data_(this->image_height_, this->image_width_, CV_8UC3,cv::Scalar(this->padding_value_, this->padding_value_, this->padding_value_));
        float scale = std::max((img[i].cols * 1.0) / this->image_width_, (img[i].rows * 1.0) / this->image_height_);
        int imgHeight_k = (int)(img[i].rows / scale);
        int imgWidth_k = (int)(img[i].cols / scale);
        cv::Mat re(imgHeight_k, imgWidth_k, CV_8UC3);
        cv::resize(img[i], re, re.size());
        // cv::cuda::resize(img, re, re.size());
        // cudaResize(img, res, re.size());
        // cv::Mat out(this->image_width_, this->image_height_, CV_8UC3, cv::Scalar(114, 114, 114));
        int reX = (this->image_width_ - re.cols) / 2;
        int reY = (this->image_height_ - re.rows) / 2;
        re.copyTo(img_data_(cv::Rect(reX, reY, re.cols, re.rows)));
        // cv::Mat img_data_;
        // cv::resize(img[i], img_data_, cv::Size(800, 800));

        cv::cvtColor(img_data_,img_data_,cv::COLOR_BGR2GRAY);
        std::vector<float> inputData_(1 * this->image_height_ * this->image_width_);
        float *dataPtr = inputData_.data();

        for (int c = 0; c < 1; ++c)
        {
            for (int h = 0; h < this->image_height_; ++h)
            {
                for (int w = 0; w < this->image_width_; ++w)
                {
                    dataPtr[c * this->image_height_ * this->image_width_ + h * this->image_width_ + w] = static_cast<float>((img_data_.at<u_char>(h, w)/255.f-0.5f)/0.5f);
                }
            }
        }
        memcpy(inputData + i * 1 * this->image_height_ * this->image_width_, dataPtr, 1 * this->image_height_ * this->image_width_ * sizeof(float));
    }
    
    return (void *)inputData;
}

int FrameStateInference::initDict(std::string dictPath){
    std::ifstream dictFile(dictPath);
    if (!dictFile.is_open()) {
        std::cerr << "无法打开字典文件 " << dictPath << std::endl;
        return -1;
    }
    std::string line;
    // 逐行读取文件内容并存入 vector
    while (std::getline(dictFile, line)) {
        word_dicts.push_back(line);
    }
    // 关闭文件
    dictFile.close();
    return 0;
}


int FrameStateInference::loadModel(const std::string &modelPath, string configPath,std::string dictPath)
{
    
    auto dictCode = initDict(dictPath);
    tensorrt_ = new TensorRT();
    std::vector<string> inputNames = {"input"};
    std::vector<string> outputNames = {"output"};
    std::vector<std::vector<int>> inputSizes = {{1, 1, this->image_height_, this->image_width_}};
    std::vector<std::vector<int>> outputSizes = {{1,this->seq_len, this->num_classes}};
    auto pt = tensorrt_->readConfig(configPath);
    maxBatchSize_ = pt.get<int>("maxBatchSize");
    int status = tensorrt_->loadModel(modelPath, inputNames, outputNames, inputSizes, outputSizes, maxBatchSize_);
    return status;
}

std::string FrameStateInference::postProcess(std::vector<float *> outputs, int batch_size){
    float* pred = outputs[0];

    std::vector<int> res;
    for(int i=0;i<seq_len;++i,pred+=num_classes){
        int idx = std::max_element(pred,pred+num_classes)-pred;
        res.push_back(idx);
    }
    std::string resultString="";
    for(int i=0;i<res.size();++i){
        if(i==0 && res[i]!=0){
            resultString+=word_dicts[res[i]];
        }
        if(i!=0 && res[i]!=0){
            if(res[i]!=res[i-1]){
                resultString+=word_dicts[res[i]];
            }
        }
    }
    return resultString;
}

int FrameStateInference::detection(cv::Mat &inputImage,std::string &output)
{
    std::vector<cv::Mat> input_imgs = {inputImage};
    auto inputData = beforeProcess(input_imgs);
    float *tmp_output = new float[maxBatchSize_ * this->seq_len*this->num_classes];
    std::vector<float *> output_ = {tmp_output};
    int status = tensorrt_->doBatchInference({(float *)inputData}, input_imgs.size(), output_);
    afterProcess(inputData);
    output = postProcess(output_,1);
    delete[] tmp_output;
    return status;
}

