import torch
import numpy as np
from transformers import AutoTokenizer
import coremltools as ct
import coremltools.models.model as ctm

# Load the tokenizer for meta-llama3-8b-Instruct
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load the .mlpackage or .mlmodelc
mlmodel = ct.models.CompiledMLModel("MetaLlama-3-8B-Instruct.mlmodelc")  # Update with your model path

# Encode the sentence fragment as input
sentence_fragment = "The Manhattan bridge is"
context = torch.tensor(tokenizer.encode(sentence_fragment)).unsqueeze(0)  # Add batch dimension

# Ensure the input is of the correct shape (1, 128)
context = torch.nn.functional.pad(context, (0, 128 - context.shape[1]), 'constant', 0)  # Padding to 128 tokens if necessary

# Run the converted Core ML model
coreml_inputs = {"input_ids": context.to(torch.int32).numpy()}
prediction_dict = mlmodel.predict(coreml_inputs)

# Handle the logits output
logits = prediction_dict["logits"]
logits_tensor = torch.tensor(logits, dtype=torch.float16)  # Convert to a tensor

# Decode the logits to get the generated text
generated_tokens = torch.argmax(logits_tensor, dim=-1).squeeze(0).tolist()
generated_text = tokenizer.decode(generated_tokens)

print("Fragment: {}".format(sentence_fragment))
print("Completed: {}".format(generated_text))
