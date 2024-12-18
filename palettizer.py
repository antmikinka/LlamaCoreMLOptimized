import coremltools as ct
import coremltools.optimize as cto


model = ct.models.MLModel("/Volumes/NVME 3/MetaLlama-3-8B-Instruct.mlpackage")

config = cto.coreml.OptimizationConfig(
	global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=8)
)
compressed_model = cto.coreml.palettize_weights(model, config)
compressed_model.save('/Volumes/NVME 3/MetaLlama-3-8B-Instruct-KMeans-8Bit.mlpackage')

