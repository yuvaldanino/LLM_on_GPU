import torch
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from transformers import AutoTokenizer, AutoModelForCausalLM

#  Step 1: Load tokenizer and PyTorch model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

#  Step 2: Tokenize input prompt
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

#  Step 3: Trace PyTorch model with dummy input
# This converts dynamic model to a static computation graph
traced_model = torch.jit.trace(model, input_ids)

#  Step 4: Convert PyTorch model to TVM Relay IR
input_name = "input_ids"
shape_dict = {input_name: input_ids.shape}
mod, params = relay.frontend.from_pytorch(traced_model, shape_dict)

#  Step 5: Compile model with TVM for CPU (use "cuda" for NVIDIA GPUs)
target = "llvm"  # "llvm" = CPU target
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

#  Step 6: Run with TVM's graph executor
dev = tvm.cpu()
module = graph_executor.GraphModule(lib["default"](dev))

# Set input and run model
module.set_input(input_name, tvm.nd.array(input_ids.numpy().astype("int32")))
module.run()

#  Step 7: Get output and decode
output = module.get_output(0).asnumpy()
token_ids = np.argmax(output, axis=-1)
decoded = tokenizer.decode(token_ids[0], skip_special_tokens=True)

print("Prompt:", prompt)
print("Generated:", decoded)
