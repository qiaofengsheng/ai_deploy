#ifndef __TENSORRT_H__
#define __TENSORRT_H__
#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // 输出日志消息
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRT
{

public:
    TensorRT();
    ~TensorRT();
    // int LoadModel(string modelPath,int maxBatchSize);

public:
    int loadModel(string modelPath, std::vector<string> inputNames,
                  std::vector<string> outputNames,
                  std::vector<std::vector<int>> inputSizes, std::vector<std::vector<int>> outputSizes, int maxBatchSize);
    int doBatchInference(std::vector<float *> input, const int batch_size, std::vector<float *> &outputs);

    int loadOnnxModel(string modelPath);
    int loadTrtModel(string modelPath);
    boost::property_tree::ptree readConfig(string configPath);

private:
    std::vector<string> inputNames;
    std::vector<string> outputNames;
    std::vector<std::vector<int>> inputSizes;
    std::vector<std::vector<int>> outputSizes;
    std::vector<int> inputVolumSize;
    std::vector<int> outputVolumSize;
    int minBatch = 1;
    int mediumBatch = 2;
    int maxBatch = 4;
    Logger logger_;
    ICudaEngine *engine_;
    IExecutionContext *context_;
    cudaStream_t stream_;
    std::vector<void *> vecBuffer_;
};

#endif