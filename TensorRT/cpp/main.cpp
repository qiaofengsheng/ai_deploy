#include"trtInference.hpp"

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}
int main(){
    TensorrtInference* t = new TensorrtInference();
    t->initModel("/data/qfs/project/study/trt_demo/python/model.engine",1);
    // t->onnxToEngine("/data/qfs/project/study/trt_demo/python/model.onnx","./model.engine",1,2,4);
    // float data1[1 * 3 * 224 * 224];
    // float data2[1 * 3 * 224 * 224];
    // for (int i = 0; i < 1 * 3 * 224 * 224; i++){
    //     data1[i] = 1.0f;
    //     data2[i] = 1.0f;
    // }
            
    
    vector<float*> data;
    // data.push_back(data1);
    // data.push_back(data2);
    cv::Mat img = cv::imread("/data/qfs/project/study/cpp_http_demo/img/3.jpg");
    cv::resize(img,img,cv::Size(224,224));
    float *data1 = blobFromImage(img);
    for(int i=0;i<img.total()*3;i++){
        cout<<data1[i]<<endl;
    }
    data.push_back(data1);
    data.push_back(data1);
    // 执行推理
    float prob1[1 * 1000];
    float prob2[1 * 2];
    vector<float*> prob;
    prob.push_back(prob1);
    prob.push_back(prob2);
    t->doInference(data,prob);
    cout<<prob[1][0]<<" "<<prob[1][1]<<endl;
    return 0;
}