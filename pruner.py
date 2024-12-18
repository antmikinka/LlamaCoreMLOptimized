import coremltools as ct
import coremltools.optimize as cto
import torch

#torch.mps.set_per_process_memory_fraction(0.3)

from coremltools.optimize.coreml import (
	OpThresholdPrunerConfig,
	OptimizationConfig,
	prune_weights,
)



model = ct.models.MLModel("MetaLlama-3-8B-Instruct.mlpackage")



op_config = OpThresholdPrunerConfig(
	threshold=0.04,
	minimum_sparsity_percentile=0.5,
	weight_threshold=1024,
)
config = OptimizationConfig(global_config=op_config)
model_compressed = prune_weights(model, config=config)
model_compressed.save("MetaLlama-3-8B-Instruct-Pruned.mlpackage")