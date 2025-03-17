import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Load GPT-2 Model from Hugging Face
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Prepare Dummy Input for ONNX Export
dummy_input = torch.randint(0, 1000, (1, 10))  # Simulates a text prompt

# ✅ Export Model to ONNX Format
onnx_path = "gpt2.onnx"
torch.onnx.export(model, dummy_input, onnx_path, 
                  input_names=["input_ids"], output_names=["output"], 
                  dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}})

print(f"✅ GPT-2 Model successfully converted to {onnx_path}")
