from typing import Union, Optional, Sequence,Dict,Any
import torch
import tensorrt as trt

class TRTWrapper(torch.nn.Module):
    def __init__(self,engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):    
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) # 反序列化

        '''
        执行上下文对象是 TensorRT 运行时的核心对象，它包含了 TensorRT 引擎的执行状态和所有的中间计算结果。
        创建执行上下文对象之后，可以将输入数据输入到 TensorRT 引擎中进行推理，并获取输出结果。
        同时，执行上下文对象还包含了一些其他的配置参数，例如 batch 大小、工作空间大小等，可以通过修改这些参数来优化模型的性能
        '''
        self.context = self.engine.create_execution_context()   # 创建一个执行上下文对象，用于进行模型推理。


        names = [_ for _ in self.engine]    # 获取 TensorRT 引擎中所有层的名称列表 ['input','output']
        print(self.engine.binding_is_input)
        input_names = list(filter(self.engine.binding_is_input, names)) # 返回值是一个包含了所有输入层名称的列表 ['input']
        self._input_names = input_names
        self._output_names = output_names

        # 如果内有指定output name，就通过所有层名称列表减去输入层名称列表得到输出层名称列表
        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names)) # 创建长度为*的列表
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name) # 通过profile_id获取指定的模型的profile信息，input_name模型输入名称
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name) # 获取engine中参数绑定的索引，用来检索和访问模型特定参数的值
            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()   # 确保内存是连续分配的
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))  # 设置当前上下文中给定索引位置的绑定形状。 idx参数是绑定索引的位置，而tuple(input_tensor.shape)将使用特定输入张量的形状创建一个元组
            bindings[idx] = input_tensor.contiguous().data_ptr()    # 将input_tensor的数据首地址赋值给bindings中idx索引处的元素

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream) # 运行 CUDA 异步操作的方法。它提供两个参数：第一个是异步操作的执行参数的集合，第二个是一个用于执行 CUDA 操作的流。
        return outputs

model = TRTWrapper('/data/qfs/project/study/trt_demo/python/model.engine', ['output1','output2'])
import time
s1 = time.time()

# for i in range(10):
#     output = model(dict(input = torch.ones(1, 3, 224, 224).cuda()))
# e1 = time.time()


# output = model(dict(input = torch.ones(16, 3, 224, 224).cuda()))
# e2 = time.time()

# output = model(dict(input = torch.ones(32, 3, 224, 224).cuda()))
# e3 = time.time()


# output = model(dict(input = torch.ones(64, 3, 224, 224).cuda()))
# e4 = time.time()
import cv2
img = cv2.imread("/data/qfs/project/study/cpp_http_demo/img/3.jpg")
img = cv2.resize(img,(224,224))
print(img)
img = torch.Tensor(img).permute(2,0,1).unsqueeze(0)
print(img)
output = model(dict(input1 = img.cuda(),input2 = img.cuda()))
e5 = time.time()
print(output["output1"])
print(output["output2"])
