# 用法
1.将导出的onnx模型放在model目录下，其中支持动态和静态batch，再config.ini设置最大batch

2.使用createModel进行模型的转换，可以直接使用onnx也可以直接使用trt模型，运行中如果是onnx会自动转为trt模型

3.使用detection(matImg, outputs)得到最终的输出结果

4.其中置信度以及nms还有宽高的设置是在include/frame_state_inference.hpp中

5.后续会进行优化，目前都是基于cpu版本的nms，可自行将之前tensort下面的gpu的nms编译进去进行加速

6.其中网络的输出顺序为[x,y,w,h,obj,class1,class2,....]
