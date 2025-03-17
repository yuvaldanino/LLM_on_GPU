import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ Load ONNX Runtime with DirectML Execution
onnx_path = "gpt2.onnx"
ort_session = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider"])

# ✅ Test Input
input_text = "Tell me a fun fact."
input_ids = tokenizer(input_text, return_tensors="np")["input_ids"].astype(np.int64)

# ✅ Run ONNX Inference
outputs = ort_session.run(None, {"input_ids": input_ids})
print("✅ ONNX Model Output:", outputs)
