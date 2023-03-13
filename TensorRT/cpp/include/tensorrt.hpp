#ifndef __TENSORRT_H__
#define __TENSORRT_H__
#include<iostream>
#include<fstream>
#include<map>
#include<vector>
#include<NvInfer.h>
#include<NvOnnxParser.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cassert>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

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

class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) override {
            // 输出日志消息
            if (severity != Severity::kINFO) {
                std::cout << msg << std::endl;
            }
        }
};

class TensorRT{

    public:
        TensorRT();
        ~TensorRT();
        // int LoadModel(string modelPath,int maxBatchSize);

    public:
        int loadModel(string modelPath,int batchSize);
        int doInference(float* input, const int batch_size, std::vector<void*>& outputs);

        int loadOnnxModel(string modelPath);
        int loadTrtModel(string modelPath,int batchSize);


    public:
        // 模型输入的尺寸
        std::vector<Dims> inputDims_;
        std::vector<int64_t> inputSizes_;
        std::vector<int> inputIndexes_;

        // 模型输出的尺寸
        std::vector<Dims> outputDims_;
        std::vector<int64_t> outputSizes_;
        std::vector<void *> outputPtrs_;
        std::vector<int> outputIndexes_;

    private:
        std::vector<string> inputNames={"input"};
        std::vector<string> outputNames={"output"};

        int minBatch = 1;
        int mediumBatch = 2;
        int maxBatch = 4;

        Logger logger_;
        ICudaEngine *engine_;
        IExecutionContext *context_;
        cudaStream_t stream_;
        void *buffers_[2];
        // void *buffer;
};

#endif