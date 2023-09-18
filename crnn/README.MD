# 用法
1.将导出的onnx模型放在model目录下，config.ini设置最大batch,dicts.txt是自定义的字典

2.使用createModel进行模型的转换，可以直接使用onnx也可以直接使用trt模型，运行中如果是onnx会自动转为trt模型

3.使用detection(inputImage,resultString)得到最终的输出结果

4.其中类别数、序列长度以及宽高的设置是在include/frame_state_inference.hpp中

5.模型导出的维度需要将batch放在第一维，形状为[batch_size,seq_len,num_classes]



