import torch
import coremltools as ct
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
import torch.nn
import torch.mps

from datasets import load_dataset

torch.mps.set_per_process_memory_fraction(0.3)

#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


compute_units = ct.ComputeUnit.ALL
compute_precision = ct.precision.FLOAT16

#torch.mps.set_per_process_memory_fraction(0.3)

class StatefulMistral(torch.nn.Module):
	def __init__(self, modelPath, batchSize=1, contextSize=64):
		super().__init__()
		self.model = LlamaForCausalLM.from_pretrained(modelPath)
		
		#self.register_buffer("keyCache", torch.zeros(self.kvCacheShape))
		#self.register_buffer("valueCache", torch.zeros(self.kvCacheShape))

	def forward(self, input_ids):
		output = self.model(input_ids)
		return output.logits


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Initialize the model
torch_model = StatefulMistral(model_id)
#torch_model = torch_model.half() I guess this doesnt reduce memory


torch_model.half() #model.half() does allow me to convert further actually. 6/9/24


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Set the EOS token as the padding token
tokenizer.pad_token_id = tokenizer.eos_token_id

## PRUNING CODE
# https://developer.apple.com/videos/play/wwdc2024/10159/
# code snippet from a WWDC video
# Prune the PyTorch model using SparseGPT with calibration data

from coremltools.optimize.torch.layerwise_compression import (
	LayerwiseCompressor, 
	LayerwiseCompressorConfig, 
	ModuleSparseGPTConfig,
)

prune_config = ModuleSparseGPTConfig(target_sparsity=0.4)
compressor_config = LayerwiseCompressorConfig(
	global_config=prune_config, 
	calibration_nsamples=128,
)

pruner = LayerwiseCompressor(torch_model, compressor_config)







def calibration_data_loader(nsamples=16, seqlen=64):
	"""
	Generator function to load and tokenize the dataset.
	Yields batches of tokenized data.
	"""
	# Load the dataset
	train_data = load_dataset("SkunkworksAI/reasoning-0.01", split="train") # Adjust the split as needed
	
	# Select a subset of the dataset
	small_train_data = train_data.select(range(min(nsamples, len(train_data))))
	
	# Yield tokenized batches
	for i in range(0, len(small_train_data), nsamples):
		batch = small_train_data[i: i + nsamples]
		tokenized_batch = tokenizer(batch['reasoning'], truncation=True, padding='max_length', max_length=seqlen, return_tensors='pt')
		yield tokenized_batch['input_ids']

'''
def loss_fn(model, data):
	"""
	Perform forward pass on the model and compute loss.
	"""
	lm_logits = model(data).logits
	shift_logits = lm_logits[:, :-1, :].contiguous()
	shift_labels = data[:, 1:].contiguous()

	loss_fct = torch.nn.CrossEntropyLoss()
	return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
'''
#i dont think i need loss_fn, as its no tin pruner.compress

# Assuming `pruner` and `model` are defined elsewhere
sparse_model = pruner.compress(calibration_data_loader(nsamples=16, seqlen=64), device=torch.device("mps"))

# RuntimeError: MPS backend out of memory (MPS allocated: 9.02 GB, other allocations: 384.00 KB, max allowed: 9.07 GB). Tried to allocate 112.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure)

#sparse_model = pruner.compress(calibration_data_loader, device = torch.device("mps")



## PALETTIZING CODE
# https://developer.apple.com/videos/play/wwdc2024/10159/
# code snipped from a WWDC video
# Palettize previously pruned PyTorch model `sparse_model`

from coremltools.optimize.torch.layerwise_compression import (
	PostTrainingPalettizer, 
	PostTrainingPalettizerConfig, 
	ModulePostTrainingPalettizerConfig,
)

palettize_config = ModulePostTrainingPalettizerConfig(
	nbits=4,
	granularity="per_grouped_channel", #may change this dependent upon bugs during / before CT conversion
	group_size=16,
)

ptp_config = ModulePostTrainingPalettizerConfig(global_config=palettize_config)

palettizer = PostTrainingPalettizer(sparse_model, ptp_config)

sparse_palettized_model = palettizer.compress()























# Example input tensors
shape = (1, 64)
input_ids = np.random.randint(0, 64, shape)
input_ids = torch.tensor(input_ids, dtype=torch.int32)

class Wrapper(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model.eval()
		
	def forward(self, input_ids):
		return self.model(
			input_ids=input_ids
		)

to_jit = Wrapper(sparse_palettized_model.eval()) ## i added this after meta-llama-3-8b-instruct conversion, as it still came up with no .eval, I do not know if it would work
to_jit.eval() #maybe this works?
with torch.no_grad():
	output_jit = to_jit(input_ids)



# Define input and output types for Core ML conversion
coreml_input_types = [ct.TensorType(
	name="input_ids",
	shape=ct.Shape(shape=shape),
	dtype=np.int32,
)]

coreml_output_types = [ct.TensorType(name="logits", dtype=np.float16)]
#coreml_output_types = [ct.TensorType(name="logits", dtype=np.float32)]
# Trace the model

traced_model = torch.jit.trace(sparse_palettized_model.eval(), [input_ids])
# changed above to sparse_palettized_model

# Convert the traced model to Core ML
fp16_stateful_mlmodel = ct.convert(
	traced_model,
	inputs=coreml_input_types,
	#states=states,
	outputs=coreml_output_types,
	minimum_deployment_target=ct.target.iOS18,
	compute_precision=compute_precision,
	compute_units=compute_units
)

fp16_stateful_mlmodel.save("MetaLlama-3-8B-Instruct-SparsePalettized.mlpackage")

# Save the converted model
#fp16_stateful_mlmodel.save("MetaLlama-3-8B-Instruct-SparsePalettized.mlpackage")


#architecture = torch_model.config.model_type
architecture = "LlamaForCasualLM"
'''
user_defined_metadata = {
	"co.huggingface.exporters.name": model_name,
	"co.huggingface.exporters.task": "text-generation",
	"co.huggingface.exporters.architecture": architecture,
	"co.huggingface.exporters.framework": "pytorch",
	"co.huggingface.exporters.precision": compute_precision,
}
'''

# Assuming `compute_precision` is used to apply float16 precision selectively
precision_description = "FP16"

user_defined_metadata = {
	"co.huggingface.exporters.name": model_id,
	"co.huggingface.exporters.task": "text-generation",
	"co.huggingface.exporters.architecture": architecture,
	"co.huggingface.exporters.framework": "pytorch",
	"co.huggingface.exporters.precision": precision_description,
}



#spec = fp16_stateful_mlmodel._spec
#spec.description.metadata.userDefined.update(user_defined_metadata)



card = f"""
This repository contains a Core ML conversion of [{model_id}](https://hf.co/{model_id}) with the following characteristics:

	- Sequence length: {128}, fixed.
	- Precision: {precision_description}.

Please, check the [original model card](https://hf.co/{model_id}) for additional details on the model.
"""
print(user_defined_metadata)
print(card)