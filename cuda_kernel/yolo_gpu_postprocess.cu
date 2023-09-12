#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <stdio.h>
__global__ void decodeGpuKernel(float *predictsGpu, float *outputGpu, float scale, float Pw, float Ph, float threshold, float nmsThreshold, int numBoxes, int numClasses, int maxObjects)
{
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= numBoxes)
        return;
    float *pitem = predictsGpu + position * (5 + numClasses);
    float objectConf = pitem[4];
    if (objectConf <= threshold)
        return;

    float *pclasses = pitem + 5;
    int label = 0;
    float classConf = pclasses[0];
    for (int i = 1; i < numClasses; i++)
    {
        if (pclasses[i] > classConf)
        {
            label = i;
            classConf = pclasses[i];
        }
    }
    float confidence = objectConf * classConf;
    if (confidence <= threshold)
        return;

    int index = atomicAdd(outputGpu, 1);
    if (index >= maxObjects)
        return;

    float cx = pitem[0];
    float cy = pitem[1];
    float w = pitem[2];
    float h = pitem[3];

    float x1 = (cx - w / 2 - Pw) / scale;
    float y1 = (cy - h / 2 - Ph) / scale;
    float x2 = (cx + w / 2 - Pw) / scale;
    float y2 = (cy + h / 2 - Ph) / scale;

    float *poutitem = outputGpu + (1 + index * 7);
    poutitem[0] = x1;
    poutitem[1] = y1;
    poutitem[2] = x2;
    poutitem[3] = y2;
    poutitem[4] = confidence;
    poutitem[5] = label;
    poutitem[6] = 0;
}

static __device__ float iou(float left1, float top1, float right1, float bottom1, float left2, float top2, float right2, float bootom2)
{
    float x1 = max(left1, left2);
    float y1 = max(top1, top2);
    float x2 = min(right1, right2);
    float y2 = min(bottom1, bootom2);
    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float area1 = (right1 - left1) * (bottom1 - top1);
    float area2 = (right2 - left2) * (bootom2 - top2);
    float area = width * height;
    return area / (area1 + area2 - area);
}

__global__ void nmsGpu(float *outputGpu, int maxObjects, float nmsThreshold)
{

    int position = blockIdx.x * blockDim.x + threadIdx.x;

    int boxCounts = min(int(*outputGpu), maxObjects);

    if (position >= boxCounts)
        return;

    // left top right bottom confidence label keep
    float *pitemNow = outputGpu + (1 + position * 7);

    for (int i = 0; i < boxCounts; i++)
    {
        float *tmpBox = outputGpu + (1 + i * 7);
        if (position == i || tmpBox[6] == 1)
            continue;
        if (pitemNow[5] != tmpBox[5])
            continue;
        if (tmpBox[4] >= pitemNow[4])
        {
            float iouValue = iou(pitemNow[0], pitemNow[1], pitemNow[2], pitemNow[3], tmpBox[0], tmpBox[1], tmpBox[2], tmpBox[3]);
            if (iouValue > nmsThreshold)
            {
                pitemNow[6] = 1;
                printf("iouValue:%f\n", iouValue);
                return;
            }
        }
    }
}

void decodeGpu(float *predictsGpu, float *outputGpu, std::vector<int> imgSize, std::vector<int> outputSize, float threshold, float nmsThreshold, int numBoxes, int numClasses, int maxObjects, cudaStream_t stream)
{
    int block = numBoxes > 512 ? 512 : numBoxes;
    int grid = (numBoxes + block - 1) / block;
    float scale = std::min(float(outputSize[0]) / float(imgSize[0]), float(outputSize[1]) / float(imgSize[1]));
    float paddingWidth = (outputSize[0] - imgSize[0] * scale) / 2;
    float paddingHeight = (outputSize[1] - imgSize[1] * scale) / 2;
    decodeGpuKernel<<<grid, block, 0, stream>>>(predictsGpu, outputGpu, scale, paddingWidth, paddingHeight, threshold, nmsThreshold, numBoxes, numClasses, maxObjects);

    int nmsblock = maxObjects > 512 ? 512 : maxObjects;
    int nmsgrid = (maxObjects + nmsblock - 1) / nmsblock;
    nmsGpu<<<nmsgrid, nmsblock, 0, stream>>>(outputGpu, maxObjects, nmsThreshold);
}