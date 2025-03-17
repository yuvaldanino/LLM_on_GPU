import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use DirectML (AMD GPU)
device = torch_directml.device()

# Load model and tokenizer (cached after first download)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")

# Test input
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Run inference
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:", generated_text)
