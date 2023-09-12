#include "frame_state_inference.hpp"

FrameStateInference::FrameStateInference()
{
}

FrameStateInference::~FrameStateInference()
{
    delete tensorrt_;
}

void *FrameStateInference::beforeProcess(std::vector<cv::Mat> &img)
{
    float *inputData = new float[img.size() * 3 * 600 * 600];
    for (int i = 0; i < img.size(); ++i)
    {
        cv::Mat img_data_;
        cv::resize(img[i], img_data_, cv::Size(600, 600));
        img_data_.convertTo(img_data_, CV_32F, 1 / 255.0);
        std::vector<float> inputData_(3 * 600 * 600);
        float *dataPtr = inputData_.data();

        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < 600; ++h)
            {
                for (int w = 0; w < 600; ++w)
                {
                    dataPtr[c * 600 * 600 + h * 600 + w] = img_data_.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        memcpy(inputData + i * 3 * 600 * 600, dataPtr, 3 * 600 * 600 * sizeof(float));
    }
    return (void *)inputData;
}

std::vector<float> softmax(const std::vector<float> inputs)
{
    std::vector<float> outputs;
    float maxVal = *std::max_element(inputs.begin(), inputs.end());

    float sum = 0.0;
    for (float input : inputs)
    {
        float val = std::exp(input - maxVal);
        outputs.push_back(val);
        sum += val;
    }

    for (float &output : outputs)
    {
        output /= sum;
    }

    return outputs;
}

int findMaxIndex(const std::vector<float> values)
{
    auto maxElement = std::max_element(values.begin(), values.end());
    int maxIndex = std::distance(values.begin(), maxElement);
    return maxIndex;
}

void FrameStateInference::postProcess(std::vector<float *> outputs, int batch_size, std::vector<Json::Value> &outputs_)
{
    float *out = (float *)outputs[0];
    for (int i = 0; i < batch_size * 2; i += 2)
    {
        std::vector<float> inputs;
        inputs.push_back(out[i]);
        inputs.push_back(out[i + 1]);
        std::vector<float> softmax_outputs = softmax(inputs);
        int maxIndex = findMaxIndex(softmax_outputs);
        Json::Value outRes;
        outRes["res"] = maxIndex;
        outRes["score"] = softmax_outputs[maxIndex];
        outputs_.push_back(outRes);
    }

    for (int i = 0; i < outputs.size(); ++i)
    {
        delete outputs[i];
    }
}

int FrameStateInference::loadModel(const std::string &modelPath, string configPath)
{
    tensorrt_ = new TensorRT();
    std::vector<string> inputNames = {"input"};
    std::vector<string> outputNames = {"output"};
    std::vector<std::vector<int>> inputSizes = {{1, 3, 600, 600}};
    std::vector<std::vector<int>> outputSizes = {{1, 2}};
    auto pt = tensorrt_->readConfig(configPath);
    maxBatchSize_ = pt.get<int>("maxBatchSize");
    int status = tensorrt_->loadModel(modelPath, inputNames, outputNames, inputSizes, outputSizes, maxBatchSize_);
    return status;
}

int FrameStateInference::detection(std::vector<cv::Mat> &input_imgs, std::vector<Json::Value> &outputs)
{
    auto inputData = beforeProcess(input_imgs);
    float *tmp_output = new float[maxBatchSize_ * 2];
    std::vector<float *> output_ = {tmp_output};
    int status = tensorrt_->doBatchInference({(float *)inputData}, input_imgs.size(), output_);
    std::cout << "output size: " << output_.size() << std::endl;
    postProcess(output_, input_imgs.size(), outputs);
    return status;
}
