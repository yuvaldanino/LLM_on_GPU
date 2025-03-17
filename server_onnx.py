import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import numpy as np


# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ Load ONNX Model for Optimized Inference
onnx_path = "gpt2.onnx"
ort_session = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider"])

# ✅ Initialize Flask API
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.json
        input_text = data.get("text", "")

        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        print("✅ Received request:", input_text)  # Debugging print

        # ✅ Tokenize input text once
        input_ids = tokenizer(input_text, return_tensors="np")["input_ids"].astype(np.int64)

        # ✅ Ensure input is exactly 2D: (batch_size, sequence_length)
        if len(input_ids.shape) == 3:  # If it's (1,1,N), flatten it to (1,N)
            input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[2])
        elif len(input_ids.shape) == 1:  # If it's just (N,), add batch dimension
            input_ids = input_ids.reshape(1, -1)

        print("✅ Converted text to input IDs:", input_ids)  # Debugging print
        print("✅ Input Shape:", input_ids.shape)  # Show the shape of the input

        outputs = ort_session.run(None, {"input_ids": input_ids})

        # ✅ Extract only the token IDs from ONNX output
        token_ids = np.argmax(outputs[0], axis=-1)  # Get highest probability token for each step

        # ✅ Flatten array if needed
        if len(token_ids.shape) > 2:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)  # Ensure shape is (1, sequence_length)

        print("✅ Token IDs from ONNX Model:", token_ids)  # Debugging print

        # ✅ Decode output
        generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        print("❌ ERROR OCCURRED:", str(e))  # Print full error message
        return jsonify({"error": str(e)}), 500


# ✅ Step 5: Start API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
