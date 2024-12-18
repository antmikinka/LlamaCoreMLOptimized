import torch
import torch.nn as nn

# Define the custom nn.Module
class ArgmaxModule(nn.Module):
	def __init__(self, axis=-1, keep_dims=False):
		super(ArgmaxModule, self).__init__()
		self.axis = axis
		self.keep_dims = keep_dims

	def forward(self, logits):
		# Only logits are used in the forward pass
		return torch.argmax(logits, dim=self.axis, keepdim=self.keep_dims)

# Initialize the module
argmax_module = ArgmaxModule()

# Example logits tensor with shape (1, 128, 128256)
logits = torch.randn(1, 128, 128256, dtype=torch.float16)

# Trace the module
traced_model = torch.jit.trace(argmax_module, logits)

# Define Core ML input and output specifications without input_ids
import coremltools as ct
import numpy as np

input_features = [ct.TensorType(name="logits", shape=(1, 128, 128256), dtype=np.float16)]
output_features = [ct.TensorType(name="output", dtype=np.int32)]  # Output shape inferred

# Convert to Core ML
fp16_stateful_mlmodel = ct.convert(
	traced_model,
	inputs=input_features,
	outputs=output_features,
	minimum_deployment_target=ct.target.iOS18,
	compute_precision=ct.precision.FLOAT16,
	compute_units=ct.ComputeUnit.ALL
)

# Save the converted Core ML model
fp16_stateful_mlmodel.save("ArgmaxModel.mlpackage")
