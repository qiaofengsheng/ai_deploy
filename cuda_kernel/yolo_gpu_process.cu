#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

__global__ void resizeCenterKernel(float *input_array, int inputW, int inputH, int channels, float *output_array, int outputW, int outputH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("x: %d, y: %d\n", x, y);

    if (x < outputW && y < outputH)
    {
        float u = (float)(x + 0.5) * ((float)inputW / (float)outputW) - 0.5f;
        float v = (float)(y + 0.5) * ((float)inputH / (float)outputH) - 0.5f;

        int x0 = (int)floor(u);
        int y0 = (int)floor(v);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float lx = u - (float)x0;
        float ly = v - (float)y0;

        for (int i = 0; i < channels; i++)
        {
            float a = input_array[(y0 * inputW + x0) * channels + i];
            float b = input_array[(y0 * inputW + x1) * channels + i];
            float c = input_array[(y1 * inputW + x0) * channels + i];
            float d = input_array[(y1 * inputW + x1) * channels + i];

            output_array[(y * outputW + x) * channels + i] = (float)((1 - ly) * ((1 - lx) * a + lx * b) + ly * ((1 - lx) * c + lx * d));
        }
    }
}

void resizeCenter(float *input_array, int inputW, int inputH, int channels, float *output_array, int outputW, int outputH)
{
    dim3 block(32, 32);
    dim3 grid((outputW + 32 - 1) / 32, (outputH + 32 - 1) / 32);
    resizeCenterKernel<<<grid, block>>>(input_array, inputW, inputH, channels, output_array, outputW, outputH);
    cudaDeviceSynchronize();
}