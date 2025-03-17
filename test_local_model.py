import torch
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Use DirectML (AMD GPU)
device = torch_directml.device()

# ✅ Load model and tokenizer from cache (pre-downloaded)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")

# ✅ Ask the model to tell a joke
input_text = "Tell me a joke."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# ✅ Run inference
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

# ✅ Decode and print joke
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n🤖 AI's Joke: ", generated_text)
