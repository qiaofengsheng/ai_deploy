#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define check(op) __checkCudaRuntime((op), #op, __FILE__, __LINE__)

bool __checkCudaRuntime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed.\n code=%s,message=%s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}
struct Box
{
    float x1;
    float y1;
    float x2;
    float y2;
    int cls;
    float conf;
};

std::vector<uint8_t> loadFile(std::string path)
{
    std::ifstream file(path, std::ios::binary);

    if (file.is_open())
    {
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> buffer(size);
        file.read((char *)buffer.data(), size);
        file.close();
        return buffer;
    }
    else
    {
        std::cout << "open file failed" << std::endl;
        return {};
    }
}

void decodeGpu(float *predictsGpu, float *outputGpu, std::vector<int> imgSize, std::vector<int> outputSize, float threshold, float nmsThreshold, int numBoxes, int numClass, int maxObjects, cudaStream_t stream);

main(int argc, char const *argv[])
{
    /* code */
    cv::Mat img = cv::imread("/home/qfs/study/tensorrt/bus.jpg");
    cv::Size imgSize = img.size();
    cv::Size outputSize = cv::Size(640, 640);
    float threshold = 0.5;
    float nmsThreshold = 0.2;
    int numClasses = 80;

    std::vector<uint8_t> result = loadFile("/home/qfs/study/tensorrt/predict.data");
    float *predicts = (float *)result.data();
    int numBoxes = result.size() / sizeof(float) / 85;
    int maxObjects = 1000;
    int numBoxElement = 7; // left,top,right,bottom,objectConf,classIndex,keepFlag
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    float *predictsGpu;
    float *outputGpu;
    float *outputHost;

    check(cudaMalloc(&predictsGpu, result.size()));
    check(cudaMemcpy(predictsGpu, predicts, result.size(), cudaMemcpyHostToDevice));
    check(cudaMalloc(&outputGpu, sizeof(float) + maxObjects * numBoxElement * sizeof(float)));
    check(cudaMallocHost(&outputHost, sizeof(float) + maxObjects * numBoxElement * sizeof(float)));

    decodeGpu(predictsGpu, outputGpu, {imgSize.width, imgSize.height}, {outputSize.width, outputSize.height}, threshold, nmsThreshold, numBoxes, numClasses, maxObjects, stream);

    check(cudaMemcpy(outputHost, outputGpu, sizeof(float) + maxObjects * numBoxElement * sizeof(float), cudaMemcpyDeviceToHost));

    int c = outputHost[0];
    for (int i = 0; i < c; ++i)
    {
        float *box = outputHost + 1 + i * numBoxElement;
        if (int(box[6]) == 1)
            continue;
        cv::rectangle(img, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), cv::Scalar(0, 0, 255), 2);
    }
    // for (auto box : boxes)
    // {
    //     cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 0, 255), 2);
    // }
    cv::imwrite("result.jpg", img);
    return 0;
}
