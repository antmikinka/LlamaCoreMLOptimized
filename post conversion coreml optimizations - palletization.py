from coremltools.optimize.coreml import (
	OpPalettizerConfig,
	OptimizationConfig,
	palettize_weights,
)

op_config = OpPalettizerConfig(mode="kmeans", nbits=6, weight_threshold=512)
config = OptimizationConfig(global_config=op_config)
compressed_6_bit_model = palettize_weights(model, config=config)