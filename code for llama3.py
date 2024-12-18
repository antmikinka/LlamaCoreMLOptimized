 import torch
 import coremltools as ct
 import transformers
 from transformers import LlamaForCausalLM, AutoTokenizer
 import numpy as np
 import torch.nn
 import torch.mps
 
 #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
 
 
 compute_units = ct.ComputeUnit.CPU_AND_NE
 compute_precision = ct.precision.FLOAT16
 
 #torch.mps.set_per_process_memory_fraction(0.3)
 
 class StatefulMistral(torch.nn.Module):
	 def __init__(self, modelPath, batchSize=1, contextSize=128):
		 super().__init__()
		 self.model = LlamaForCausalLM.from_pretrained(modelPath)
		 
		 #self.register_buffer("keyCache", torch.zeros(self.kvCacheShape))
		 #self.register_buffer("valueCache", torch.zeros(self.kvCacheShape))
 
	 def forward(self, input_ids):
		 output = self.model(input_ids)
		 return output.logits
 
 # Initialize the model
 torch_model = StatefulMistral("meta-llama/Meta-Llama-3-8B-Instruct")
 torch_model = torch_model.half()
 
 # Example input tensors
 shape = (1, 128)
 input_ids = np.random.randint(0, 128, shape)
 input_ids = torch.tensor(input_ids, dtype=torch.int32)
 
 class Wrapper(torch.nn.Module):
	 def __init__(self, model):
		 super().__init__()
		 self.model = model.eval()
		 
	 def forward(self, input_ids):
		 return self.model(
			 input_ids=input_ids
		 )
 
 to_jit = Wrapper(torch_model.eval())
 
 with torch.no_grad():
	 output_jit = to_jit(input_ids)
 
 
 
 # Define input and output types for Core ML conversion
 coreml_input_types = [ct.TensorType(
	 name="input_ids",
	 shape=ct.Shape(shape=shape),
	 dtype=np.int32,
 )]
 
 coreml_output_types = [ct.TensorType(name="logits", dtype=np.float16)]
 
 # Trace the model
 traced_model = torch.jit.trace(to_jit.eval(), [input_ids])
 
 # Convert the traced model to Core ML
 fp16_stateful_mlmodel = ct.convert(
	 traced_model,
	 inputs=coreml_input_types,
	 outputs=coreml_output_types,
	 minimum_deployment_target=ct.target.iOS18,
	 compute_precision=compute_precision,
	 compute_units=compute_units
 )
 
 # Save the converted model
 fp16_stateful_mlmodel.save("MetaLlama-3-8B-Instruct.mlpackage")
 
 
