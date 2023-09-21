#include "frame_state_inference.hpp"

FrameStateInference::FrameStateInference()
{
    anchors_ = this->image_height_/8 * this->image_width_/8+this->image_height_/16 * this->image_width_/16+this->image_height_/32 * this->image_width_/32;
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
        img_data_.convertTo(img_data_, CV_32F, 1 / 1.0);
        std::vector<float> inputData_(3 * this->image_height_ * this->image_width_);
        float *dataPtr = inputData_.data();

        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < this->image_height_; ++h)
            {
                for (int w = 0; w < this->image_width_; ++w)
                {
                    dataPtr[c * this->image_height_ * this->image_width_ + h * this->image_width_ + w] = float(img_data_.at<cv::Vec3f>(h, w)[c])/255.0;
                }
            }
        }
        memcpy(inputData + i * 3 * this->image_height_ * this->image_width_, dataPtr, 3 * this->image_height_ * this->image_width_ * sizeof(float));
    }
    
    return (void *)inputData;
}

// 计算矩形框的面积
float calcRectArea(const cv::Rect_<float> &rect)
{
    return rect.width * rect.height;
}

// 计算矩形框的交集面积
float calcIntersectionArea(const cv::Rect_<float> &rect1, const cv::Rect_<float> &rect2)
{
    float x1 = std::max(rect1.x, rect2.x);
    float y1 = std::max(rect1.y, rect2.y);
    float x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    float y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x1 >= x2 || y1 >= y2)
        return 0.0;

    return (x2 - x1) * (y2 - y1);
}
std::vector<Instance> FrameStateInference::minNMS(std::vector<Instance> &instances, float nms_threshold)
{
    /*
        使用阈值进行筛选
    */

    std::vector<Instance> filteredInstances;
    std::vector<bool> keep(instances.size(), true);

    // 按照areas降序对实例进行排序
    std::sort(instances.begin(), instances.end(), [&](const Instance &a, const Instance &b)
                { return (a.rect.width * a.rect.height) > (b.rect.width * b.rect.height); });

    // for (auto instance : instances)
    // {
    //     std::cout << instance.prob<<" " <<instance.rect.height * instance.rect.width << " " << instance.label << " " << instance.rect.x << " " << instance.rect.y << " " << instance.rect.width << " " << instance.rect.height << std::endl;
    // }
    // 对每个实例进行NMS处理
    for (int i = 0; i < instances.size(); ++i)
    {

        if (!keep[i] || instances[i].prob < this->conf_thr_)
            continue;

        const auto &instance1 = instances[i];
        // std::cout<<instance1.label<<std::endl;
        
        filteredInstances.push_back(instance1);

        for (int j = i + 1; j < instances.size(); ++j)
        {
            if (!keep[j] || instances[j].prob < this->conf_thr_)
                continue;

            const auto &instance2 = instances[j];

            // 如果两个实例的标签（label）不同，则跳过
            if (instance1.label != instance2.label)
                continue;

            // 计算交集面积和最小面积
            float intersectionArea = calcIntersectionArea(instance1.rect, instance2.rect);
            float minArea = std::min(calcRectArea(instance1.rect), calcRectArea(instance2.rect));

            // 计算交集除以最小面积的比值
            float intersectionOverArea = intersectionArea / minArea;
            
            // 如果比值大于阈值，则将第二个实例过滤掉
            if (intersectionOverArea > nms_threshold)
            {
                keep[j] = false;
            }
        }
    }

    return filteredInstances;
}

float intersection_area(const Instance a, const Instance b)
{
    cv::Rect a_rect = a.rect;
    cv::Rect b_rect = b.rect;

    cv::Rect_<float> inter = a_rect & b_rect;
    return inter.area();
}

void FrameStateInference::NMS(const std::vector<Instance> &instances, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    int n = instances.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = instances[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Instance &a = instances[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            
            const Instance &b = instances[picked[j]];

            // 如果两个实例的标签（label）不同，则跳过
            if (a.label != b.label)
                continue;
            // intersection over union
            float inter_area = intersection_area(a, b);
            // float inter_area = intersection_rotate_area(a, b);
            // std::cout << "inter_area: " << inter_area << std::endl;
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;
            // std::cout<<"iou = "<<IoU<<" "<<a.prob<<" "<<b.prob<<""<<a.rect<<" "<<b.rect<<std::endl;
            if (inter_area / std::max(union_area, 1.f) > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void FrameStateInference::sortInstances(std::vector<Instance> &instances, int left, int right)
{
    int i = left;
    int j = right;
    float p = instances[(left + right) / 2].prob;
    while (i <= j)
    {
        while (instances[i].prob > p)
            i++;

        while (instances[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(instances[i], instances[j]);
            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j)
                sortInstances(instances, left, j);
        }
        // #pragma omp section
        {
            if (i < right)
                sortInstances(instances, i, right);
        }
    }
}

void FrameStateInference::sortInstances(std::vector<Instance> &objects)
{
    if (objects.empty())
    {
        return;
    }
    sortInstances(objects, 0, objects.size() - 1);
}


int findMaxIndex(const std::vector<float> values)
{
    auto maxElement = std::max_element(values.begin(), values.end());
    int maxIndex = std::distance(values.begin(), maxElement);
    return maxIndex;
}



int FrameStateInference::loadModel(const std::string &modelPath, string configPath)
{
    tensorrt_ = new TensorRT();
    std::vector<string> inputNames = {"images"};
    std::vector<string> outputNames = {"output0","output1"};
    std::vector<std::vector<int>> inputSizes = {{1, 3, this->image_height_, this->image_width_}};
    std::vector<std::vector<int>> outputSizes = {{1, this->anchors_, this->num_classes+4},{1,this->anchors_,int(this->image_height_/4),int(this->image_width_/4)}};
    auto pt = tensorrt_->readConfig(configPath);
    maxBatchSize_ = pt.get<int>("maxBatchSize");
    int status = tensorrt_->loadModel(modelPath, inputNames, outputNames, inputSizes, outputSizes, maxBatchSize_);
    return status;
}

void FrameStateInference::postprocess(std::vector<Instance> instances,std::vector<outStruct> &resultString){
    for (auto obj : instances)
    {
        outStruct out;
        out.x1 = int(obj.rect.x);
        out.y1 = int(obj.rect.y);
        out.x2 = int(obj.rect.x+obj.rect.width);
        out.y2 = int(obj.rect.y+obj.rect.height);
        out.conf = obj.prob;
        out.cls = obj.label;
        out.mask=obj.mask;
        resultString.push_back(out);
    }
}

cv::Mat FrameStateInference::maskProcess(Instance instance,int add_w,int add_h,float scale,cv::Size sourceSize){
    cv::Mat outMask = instance.mask(cv::Rect(int(add_w/4),int(add_h/4),int((this->image_width_-add_w*2)/4),int((this->image_width_-add_w*2)/4)));
    int lw = outMask.cols;
    int lh = outMask.rows;
    cv::resize(outMask,outMask,sourceSize,cv::INTER_NEAREST);
    // 遍历图像的像素并将矩形 roi 外的像素值设置为0
    for (int y = 0; y < outMask.rows; y++) {
        for (int x = 0; x < outMask.cols; x++) {
            if (!instance.rect.contains(cv::Point(x, y))) {
                outMask.at<uchar>(y, x) = 0;
            }
        }
    }
    return outMask;
}

int FrameStateInference::detection(cv::Mat &inputImage, std::vector<outStruct> &outputs)
{
    std::vector<cv::Mat> input_imgs = {inputImage};
    auto inputData = beforeProcess(input_imgs);
    float *tmp_output1 = new float[maxBatchSize_ * this->anchors_*(this->num_classes+4)];
    float *tmp_output2 = new float[maxBatchSize_ * this->anchors_*int(this->image_height_/4*this->image_width_/4)];
    std::vector<float *> output_ = {tmp_output1,tmp_output2};
    int status = tensorrt_->doBatchInference({(float *)inputData}, input_imgs.size(), output_);
    afterProcess(inputData);
    std::vector<Instance> proposals;
    generateDetInstances(tmp_output1,tmp_output2,this->conf_thr_, proposals);
    sortInstances(proposals);
    std::vector<int> valid_inds;
    NMS(proposals, valid_inds, this->nms_thr_);
    
    std::vector<Instance> instances;
    int count = valid_inds.size();
    instances.resize(count);
    int img_w = input_imgs[0].cols;
    int img_h = input_imgs[0].rows;
    float scale_resize = std::max((input_imgs[0].cols * 1.0) / this->image_width_, (input_imgs[0].rows * 1.0) / this->image_height_);
    float scale = std::min(this->image_width_ / (input_imgs[0].cols * 1.0), this->image_height_ / (input_imgs[0].rows * 1.0));
    int tmpH = (int)(img_h / scale_resize);
    int tmpW = (int)(img_w / scale_resize);
    int add_w = (this->image_height_ - tmpW) / 2;
    int add_h = (this->image_width_ - tmpH) / 2;

    for (int i = 0; i < count; i++)
    {
        instances[i] = proposals[valid_inds[i]];
        float x0 = (instances[i].rect.x-add_w)/scale;
        float y0 = (instances[i].rect.y-add_h)/scale;
        float x1 = (instances[i].rect.x + instances[i].rect.width-add_w)/scale;
        float y1 = (instances[i].rect.y + instances[i].rect.height-add_h)/scale;
        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        instances[i].rect.x = x0;
        instances[i].rect.y = y0;
        instances[i].rect.width = x1 - x0;
        instances[i].rect.height = y1 - y0;
        instances[i].mask = maskProcess(instances[i],add_w,add_h,scale,inputImage.size());
    }
    // instances = minNMS(instances, this->min_thr_);
    postprocess(instances,outputs);
    delete[] tmp_output1;
    delete[] tmp_output2;
    return status;
}

void FrameStateInference::generateDetInstances(float *feat_blob,float *feat_blob2, float prob_threshold, std::vector<Instance> &objects)
{
    const int num_anchors = anchors_;
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        int basic_pos = anchor_idx * (num_classes + 4);
        int mask_pos = anchor_idx*int(this->image_height_/4*this->image_width_/4);
        float x_center = feat_blob[basic_pos + 0];
        float y_center = feat_blob[basic_pos + 1];
        float w = feat_blob[basic_pos + 2];
        float h = feat_blob[basic_pos + 3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;
        int max_index=0;
        float max_value = 0;
        for(int j=0;j<this->num_classes;++j){
            if(feat_blob[basic_pos+4+j]>max_value){
                max_index=j;
                max_value = feat_blob[basic_pos+4+j];
            }
        }
        float box_prob = max_value;
        if(box_prob>this->conf_thr_){
            Instance obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = max_index;
            obj.prob = box_prob;
            cv::Mat tmpMask(int(this->image_height_/4),int(this->image_width_/4),CV_8U);
            uint8_t* mats = tmpMask.ptr<uint8_t>(0);
            for(int m_=0;m_<int(this->image_height_/4*this->image_width_/4);++m_,mats++){
                if(feat_blob2[mask_pos+m_]>=this->mask_thr_){
                    // std::cout<<feat_blob2[mask_pos+m_]<<",";
                    *mats = 1;
                }else{
                    *mats = 0;
                }
            }
            obj.mask = tmpMask;
            objects.push_back(obj);
        }
    } // class loop
}     // point anchor loop
