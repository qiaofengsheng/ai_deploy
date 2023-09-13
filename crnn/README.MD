# 用法
1.将导出的onnx模型放在model目录下，其中支持动态和静态batch，再config.ini设置最大batch

2.使用createModel进行模型的转换，可以直接使用onnx也可以直接使用trt模型，运行中如果是onnx会自动转为trt模型

3.使用detection(inputImage,predScore,predMat)得到最终的输出结果

4.其中类别数以及宽高的设置是在include/frame_state_inference.hpp中

5.模型导出的维度需要将类别维度放在最后一维，形状为[batch_size,height,width,num_classes]



