import torch
import onnx
from torchvision import models
from torch import nn
import onnxruntime
import numpy as np
import onnxsim

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.mobilenet_v2(pretrained=True)
        self.out2 = nn.Linear(1000,2)
    def forward(self, x,x2):
        out1 = self.net(x)
        out2 = self.net(x2)
        return out1,self.out2(out2)


def export_onnx(onnx_model_name,net,input_dicts:dict,output_name:list,dynamic=True,simplify=True):
    '''
        input_dicts = {"input1":(1,3,224,224),"input2":(1,3,224,224),...}
        output_name = {"output1","output2",...}

    '''
    inp = (torch.randn(i) for i in input_dicts.values())
    net.eval()
    # generate ONNX model
    input_name = [i for i in input_dicts]
    all_names = input_name+output_name
    dynamic_axes={}
    for i in all_names:
        dynamic_axes[i]={0:'batch_size'} 
    
    # Export the model
    torch.onnx.export(net,               # model being run
                    tuple(inp),                         # model input (or a tuple for multiple inputs)
                    onnx_model_name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    #   do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = [i for i in input_name],   # the model's input names
                    output_names = [i for i in output_name], # the model's output names
                    dynamic_axes=dynamic_axes if dynamic else None
                    )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)

    # 验证精度差异
    ort_session = onnxruntime.InferenceSession(onnx_model_name)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {i.name:to_numpy(torch.randn(input_dicts[i.name])) for i in ort_session.get_inputs()}
    ort_outs = ort_session.run(None, ort_inputs)
    net_input=[torch.tensor(i) for i in ort_inputs.values()]
    torch_out = net(*net_input)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # Simplify
    if simplify:
        try:
            model_onnx, check = onnxsim.simplify(onnx_model_name)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_model_name)
        except Exception as e:
            print(f'simplifier failure: {e}')


net = NaiveModel()
input_dicts = {
    "input1":(1,3,224,224),
    "input2":(1,3,224,224),
}
export_onnx(onnx_model_name = 'model.onnx',net = net,input_dicts=input_dicts,output_name=['output1','output2'],dynamic=True,simplify=True)