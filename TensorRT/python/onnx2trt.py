import tensorrt as trt
import onnx
import torch
import os

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


onnx2trt("/data/qfs/project/study/trt_demo/python/model.onnx","/data/qfs/project/study/trt_demo/python/",1,2,4)