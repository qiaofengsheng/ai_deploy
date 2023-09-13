#include "frame_state_inference.hpp"

FrameStateInference::FrameStateInference()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0,255);
    for(int c=0;c<num_classes;++c){
        cv::Vec3b color(dis(gen),dis(gen),dis(gen));
        this->colors.push_back(color);
    }
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

    float *inputData = new float[img.size() * 3 * this->image_height_ * this->image_width_];

    for (int i = 0; i < img.size(); ++i)
    {
        cv::cvtColor(img[i],img[i],cv::COLOR_BGR2RGB);
        cv::Mat img_data_(this->image_height_,this->image_width_, CV_8UC3,cv::Scalar(this->padding_value_, this->padding_value_, this->padding_value_));
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

        img_data_.convertTo(img_data_, CV_32F, 1 / 255.0);
        std::vector<float> inputData_(3 * this->image_height_ * this->image_width_);
        float *dataPtr = inputData_.data();

        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < this->image_height_; ++h)
            {
                for (int w = 0; w < this->image_width_; ++w)
                {
                    dataPtr[c * this->image_height_ * this->image_width_ + h * this->image_width_ + w] = img_data_.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        memcpy(inputData + i * 3 * this->image_height_ * this->image_width_, dataPtr, 3 * this->image_height_ * this->image_width_ * sizeof(float));
    }
    
    return (void *)inputData;
}


int FrameStateInference::loadModel(const std::string &modelPath, string configPath)
{
    tensorrt_ = new TensorRT();
    std::vector<string> inputNames = {"input"};
    std::vector<string> outputNames = {"output"};
    std::vector<std::vector<int>> inputSizes = {{1, 3, this->image_height_, this->image_width_}};
    std::vector<std::vector<int>> outputSizes = {{1,this->image_height_,this->image_width_, this->num_classes}};
    auto pt = tensorrt_->readConfig(configPath);
    maxBatchSize_ = pt.get<int>("maxBatchSize");
    int status = tensorrt_->loadModel(modelPath, inputNames, outputNames, inputSizes, outputSizes, maxBatchSize_);
    return status;
}

std::vector<cv::Mat> FrameStateInference::postProcess(std::vector<float *> outputs, int batch_size){

    cv::Mat outScore(output_height,output_width,CV_32F);
    cv::Mat outMat(output_height,output_width,CV_8U);
    float* pred = outputs[0];
    uint8_t* mats = outMat.ptr<uint8_t>(0);
    float* score = outScore.ptr<float>(0);

    for(int i=0;i<output_height*output_width;++i,pred+=num_classes,++mats,++score){
        int idx = std::max_element(pred,pred+num_classes)-pred;
        *mats =idx;
        *score =pred[idx];
    }
    return {outMat,outScore};
}

int FrameStateInference::detection(cv::Mat &inputImage,cv::Mat &predScore,cv::Mat &resultMat)
{
    std::vector<cv::Mat> input_imgs = {inputImage};
    auto inputData = beforeProcess(input_imgs);
    float *tmp_output = new float[maxBatchSize_ * this->output_height*output_width*this->num_classes];
    std::vector<float *> output_ = {tmp_output};
    int status = tensorrt_->doBatchInference({(float *)inputData}, input_imgs.size(), output_);
    afterProcess(inputData);

    std::vector<cv::Mat> outputMats = postProcess(output_,1);
    int img_w = input_imgs[0].cols;
    int img_h = input_imgs[0].rows;
    float scale_resize = std::max((input_imgs[0].cols * 1.0) / this->image_width_, (input_imgs[0].rows * 1.0) / this->image_height_);
    int tmpH = (int)(img_h / scale_resize);
    int tmpW = (int)(img_w / scale_resize);

    int add_w = (this->image_height_ - tmpW) / 2;
    int add_h = (this->image_width_ - tmpH) / 2;


    
    predScore = outputMats[1](cv::Rect(add_w,add_h,tmpW,tmpH));
    cv::Mat predMat = outputMats[0](cv::Rect(add_w,add_h,tmpW,tmpH));
    cv::resize(predMat,predMat,input_imgs[0].size(),cv::INTER_NEAREST);
    cv::resize(predScore,predScore,input_imgs[0].size(),cv::INTER_LINEAR);

    for (int i = 0; i < predMat.rows; ++i) {
        for (int j = 0; j < predMat.cols; ++j) {
            // 获取当前像素值
            int pixelValue = static_cast<int>(predMat.at<u_char>(i, j));
            if (pixelValue==0){
                continue;
            }else{
                resultMat.at<cv::Vec3b>(i,j)=colors[pixelValue-1];
            }
        }
    }



    delete[] tmp_output;
    return status;
}

