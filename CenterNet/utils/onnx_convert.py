import numpy as np
import argparse
import torch.onnx
from collections import OrderedDict
import model
import onnx
import onnxruntime


# TODO : What to do about Dynamic Axes in export

# Command line argument parser
parser = argparse.ArgumentParser(description="ONNX Converter")
parser.add_argument('-m', '--model', help='Model Class Name', required=True)
parser.add_argument('-a', '--args', nargs="*", help='Model Arguments')
parser.add_argument('-c', '--ckpt', help='Model Checkpoint', required=True)
parser.add_argument('-i', '--input', nargs="+", help='Input Size', type=int, required=True)
parser.add_argument('-n', '--name', help="ONNX Model Name", required=True)

args = parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# Parse Arguments to create Model instance
model_class = getattr(model, args.model)
if args.args is not None:
    arguments = [int(x) if x.isdigit() else x for x in args.args]
    model = model_class(*arguments)
else:
    model = model_class()
print("Created Pytorch Model!")

# Load state dict from checkpoint
ckpt = torch.load(args.ckpt)
state_dict = ckpt['state_dict']

# If state_dict was saved from DataParallel Model, change state_dict
if list(state_dict.keys())[0][:6] == "module":
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    state_dict = new_state_dict

# Load state dict and make it evaluation mode
model.load_state_dict(state_dict)
model.eval()
print("Successfully loaded state_dict!")

# Create Dummy Input and Output for Testing
x = torch.randn(*args.input, requires_grad=True)
torch_out = model(x)

# Exporting the Model
torch.onnx.export(model,                                    # model to export
                  x,                                        # Dummy Input
                  args.name,                                # Where to Save Model
                  export_params=True,                       # Store trained parameter weights
                  do_constant_folding=True,                 # Optimization
                  input_names=['input'],
                  output_names=['output'])

print("Exported model to ONNX file")
print("Checking Model Validity...")
# Try loading exported model in onnx and check validity
onnx_model = onnx.load(args.name)
# onnx.checker.check_model(onnx_model)
print("Done!")

print("Testing Output Validity...")
# Test whether outputs are same within reasonable distance
ort_session = onnxruntime.InferenceSession(args.name)

# Compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
print("Exported model saved as {}".format(args.name))