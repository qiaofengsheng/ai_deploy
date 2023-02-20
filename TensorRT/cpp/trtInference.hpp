#ifndef __TRTINFERENCE_H__
#define __TRTINFERENCE_H__

#include<iostream>
#include<fstream>
#include<map>
#include<vector>
#include<NvInfer.h>
#include<NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include"logger.h"
#include"json.h"


using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace sample;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class TensorrtInference{
    public:
        TensorrtInference();
        ~TensorrtInference();
        int initModel(string engine_path,int batch);
        int onnxToEngine(const char* onnx_path,string save_engine_path,int min_batch,int medium_batch,int max_batch);
        // void infer(cv::Mat img,string &result);
        int doInference(vector<float*> input,vector<float*> &output);


    private:
        float* blobFromImage(cv::Mat& img);
        // void dataProcess(cv::Mat img,vector<flaot*> input);
        // int doInference(vector<float*> input,vector<float*> &output);
        // void postProcess(vector<float*> output,Json::Value &result);
    private:
        map<string,vector<int>> input_names = {
            {"input1",{224,224}},
            {"input2",{224,224}}
        };
        vector<string> output_names = {"output1","output2"};
        int batch_size=1;
        static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        Logger logger;
        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;

        cudaStream_t stream;
        void* buffers[4];
};


#endif
