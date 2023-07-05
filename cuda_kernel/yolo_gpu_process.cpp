#include <cuda_runtime.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

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

void resizeCenter(float *input_array, int inputW, int inputH, int channels, float *output_array, int outputW, int outputH);

int main(int argc, char const *argv[])
{
    /* code */
    check(cudaSetDevice(0));
    cv::Mat imgSource = cv::imread("/home/qfs/study/tensorrt/548deccd9c187c82a71cc841e6b37194.jpeg");
    cv::Mat img;
    imgSource.convertTo(img, CV_32FC3);
    int inputW = img.cols;
    int inputH = img.rows;
    int channels = img.channels();
    printf("inputW = %d, inputH = %d, channels = %d\n", inputW, inputH, channels);
    int outputW = 640;
    int outputH = 640;
    float *inputDevice;
    float *outputDevice;
    float resizeScale = std::min((float)outputW / (float)inputW, (float)outputH / (float)inputH);
    int resizeW = inputW * resizeScale;
    int resizeH = inputH * resizeScale;
    float *outputHost = new float[resizeW * resizeH * channels];

    printf("resizeW = %d, resizeH = %d\n", resizeW, resizeH);
    cv::Mat outputImg(resizeH, resizeW, CV_32FC3);
    check(cudaMalloc(&inputDevice, sizeof(float) * inputW * inputH * channels));
    check(cudaMalloc(&outputDevice, sizeof(float) * resizeW * resizeH * channels));

    check(cudaMemcpy(inputDevice, img.data, sizeof(float) * inputW * inputH * channels, cudaMemcpyHostToDevice));

    resizeCenter(inputDevice, inputW, inputH, channels, outputDevice, resizeW, resizeH);
    check(cudaMemcpy(outputImg.data, outputDevice, sizeof(float) * resizeW * resizeH * channels, cudaMemcpyDeviceToHost));
    cv::Mat outResultImg(outputH, outputW, CV_32FC3, cv::Scalar(255, 255, 255));
    outputImg.copyTo(outResultImg(cv::Rect(int((outputW - outputImg.cols) / 2), int((outputH - outputImg.rows) / 2), resizeW, resizeH)));
    check(cudafree(inputDevice));
    check(cudafree(outputDevice));
    cv::imwrite("output.jpg", outResultImg);

    return 0;
}
