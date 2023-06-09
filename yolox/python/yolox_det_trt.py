"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import onnx
from yolox_utils import *
import math

CONF_THRESH = 0.003
IOU_THRESHOLD = 0.42
LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38
LEN_ALL_RESULT = 30492 * 11
LEN_ONE_RESULT = 11
LEN_ALL_RESULT = 8400 * 85
LEN_ONE_RESULT = 85


def onnx2trt(onnx_model_path,save_engine_path,min_batch,medium_batch,max_batch):
    if not os.path.exists(save_engine_path):
        os.makedirs(save_engine_path)
    names = os.path.basename(onnx_model_path)
    # 加载onnx模型
    onnx_model = onnx.load(onnx_model_path)
    input_nodes = onnx_model.graph.input        
    device = torch.device('cuda:0')
    
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR) # 记录错误信息日志
    builder = trt.Builder(logger)   # 创建模型生成器
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)   # 定义EXPLICIT_BATCH，该常量在创建网络时用于启用显示批处理模式
    network = builder.create_network(EXPLICIT_BATCH)
    # 解析 onnx
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config() # 创建tensorrt引擎
    config.max_workspace_size = 2<<20   # TensorRT 引擎在构建时可以使用的最大工作空间大小，设置为 2^{20}，即 1MB
    profile = builder.create_optimization_profile() # 创建优化配置对象 profile，用于设置 TensorRT 引擎优化时的各种选项。

    for node in input_nodes:
        s_d = node.type.tensor_type.shape.dim
        profile.set_shape(
            node.name,
            [min_batch,s_d[1].dim_value ,s_d[2].dim_value ,s_d[3].dim_value],
            [medium_batch,s_d[1].dim_value ,s_d[2].dim_value ,s_d[3].dim_value], 
            [max_batch,s_d[1].dim_value ,s_d[2].dim_value ,s_d[3].dim_value])    # 设置最小、中间、最大batch
    config.add_optimization_profile(profile)

    # 构建 engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)  
        
    with open(os.path.join(save_engine_path,names.replace("onnx","engine")), mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

class YoLoxTRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    
    def infer(self, raw_image):
                # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        np.copyto(batch_input_image[0], raw_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        return host_outputs

    def out2json(self,dets):
        jsondata = {}
        jsondata["data"] = {}

        if dets is None:
            return jsondata

        data = []
        for det in dets:
            quadrangle_ = [{
                "x": det[0],
                "y": det[1]
            }, {
                "x": det[2],
                "y": det[3]
            }, {
                "x": det[4],
                "y": det[5]
            }, {
                "x": det[6],
                "y": det[7]
            }]
            confidence_ = det[8]
            class_ = int(det[9]) + 1
            data.append({"quadrangle": quadrangle_, "confidence": confidence_, "class": class_})

        jsondata["data"] = data
        jsondata["code"] = 20000
        jsondata["msg"] = "ok"
        jsondata["requestId"] = "123456789"

        return jsondata 
    
    # def obb2poly(self,obboxes):
    #     center, w, h, theta = (obboxes[:, :2]+obboxes[:,2:4])/2, obboxes[:, ], obboxes[:, 3], obboxes[:, 4]
    #     Cos, Sin = np.cos(theta), torch.sin(theta)

    #     vector1 = torch.cat(
    #         [w/2 * Cos, -w/2 * Sin], dim=-1)
    #     vector2 = torch.cat(
    #         [-h/2 * Sin, -h/2 * Cos], dim=-1)

    #     point1 = center + vector1 + vector2
    #     point2 = center + vector1 - vector2
    #     point3 = center - vector1 - vector2
    #     point4 = center - vector1 + vector2
    #     return torch.cat(
    #         [point1, point2, point3, point4], dim=-1)

    def yolox_infer(self, image):
        from yolox_utils import preprocess
        from yolox_utils import COCO_CLASSES
        from yolox_utils import multiclass_nms, demo_postprocess, vis
        input_shape = (800, 800)
        img, ratio,(add_h,add_w) = preprocess(image, input_shape)

        output = self.infer(img)
        output[0] = np.reshape(output[0], (-1, 25))
        predictions = demo_postprocess(output[0], input_shape)
        
        boxes = predictions[:, :4]
        scores = predictions[:, 5:6] * predictions[:, 6:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy[:, 0] = boxes_xyxy[:, 0]-add_w
        boxes_xyxy[:, 1] = boxes_xyxy[:, 1]-add_h
        boxes_xyxy[:, 2] = boxes_xyxy[:, 2]-add_w
        boxes_xyxy[:, 3] = boxes_xyxy[:, 3]-add_h
        print(add_h,add_w)
        boxes_xyxy /= ratio
        # [x,y,x,y,angle,conf,cls]
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.3)
        
        if dets is not None:
            new_dets = np.zeros((dets.shape[0], 10))        
            # 暂时使用矩形框进行模拟多边形输出
            new_dets[:, 0] = dets[:, 0]
            new_dets[:, 1] = dets[:, 1]
            new_dets[:, 2] = dets[:, 2]
            new_dets[:, 3] = dets[:, 1]
            new_dets[:, 4] = dets[:, 2]
            new_dets[:, 5] = dets[:, 3]
            new_dets[:, 6] = dets[:, 0]
            new_dets[:, 7] = dets[:, 3]
            new_dets[:, 8] = dets[:, 4]
            new_dets[:, 9] = dets[:, 5]
        else:
            new_dets = None
        output = self.out2json(new_dets)

        return output
    
    
    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
    
# INFER = YoLoxTRT("yolox_s.engine")
# img = cv2.imread("/home/qfs/project/server_project/xuexiji_pipeline/images/dog.jpg")
# dets = INFER.yolox_infer(img)
# if dets is not None:
#     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
#     origin_img = vis(img, final_boxes, final_scores, final_cls_inds,
#                     conf=0.5, class_names=COCO_CLASSES)
#     cv2.imwrite("result.jpg", origin_img)
# INFER.destroy()


