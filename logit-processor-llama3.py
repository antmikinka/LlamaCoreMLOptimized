import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

# Define the custom nn.Module
class ArgmaxModule(nn.Module):
	def __init__(self, axis=-1, keep_dims=False):
		super(ArgmaxModule, self).__init__()
		self.axis = axis
		self.keep_dims = keep_dims

	def forward(self, input_ids, logits):
		return torch.argmax(logits, dim=self.axis, keepdim=self.keep_dims)

# Initialize the module and prepare inputs
argmax_module = ArgmaxModule()
input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.int32)
logits = torch.randn(1, 128, 128256, dtype=torch.float16)

# Trace the module
traced_model = torch.jit.trace(argmax_module, (input_ids, logits))
traced_model.eval()

# Define Core ML input and output specifications without specifying output shape
input_features = [ct.TensorType(name="input_ids", dtype=np.int32, shape=(1, 128)),
				  ct.TensorType(name="logits", dtype=np.float16, shape=(1, 128, 128256))]
output_features = [ct.TensorType(name="output", dtype=np.int32)]  # Remove shape specification

# Convert to Core ML
fp16_stateful_mlmodel = ct.convert(
	traced_model,
	inputs=input_features,
	outputs=output_features,
	minimum_deployment_target=ct.target.iOS18,
	compute_precision=ct.precision.FLOAT16,
	compute_units=ct.ComputeUnit.ALL
)


fp16_stateful_mlmodel.save("MetaLlama-3-8B-Instruct-logit-processor.mlpackage")
