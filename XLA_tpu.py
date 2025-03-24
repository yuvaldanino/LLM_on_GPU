import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Move model to XLA device (TPU/GPU)
device = xm.xla_device()
model = model.to(device)
model.eval()

# Step 3: Tokenize input and move to XLA
prompt = "Tell me something interesting about space."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Step 4: Generate output on XLA-accelerated hardware
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

# Step 5: Decode and print result
generated = tokenizer.decode(output[0], skip_special_tokens=True)
print("Prompt:", prompt)
print("Generated:", generated)
