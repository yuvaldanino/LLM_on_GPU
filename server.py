import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify

# ✅ Step 1: Set up DirectML (AMD GPU)
device = torch_directml.device()

# ✅ Step 2: Load Model & Tokenizer (Cached)
# better model but needs premission model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Switch to a better model
# Alternative: model_name = "microsoft/phi-2"
#model_name = "microsoft/phi-1_5"  # Good balance of size and quality
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")

# ✅ Step 3: Initialize Flask App
app = Flask(__name__)

# ✅ Step 4: Define the API Route
@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Step 4.1: Get input data from request
        data = request.json
        input_text = data.get("text", "")

        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        # Step 4.2: Convert input text to model input format
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Step 4.3: Run inference (generate text)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=100)

        # Step 4.4: Decode output and return it
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Step 5: Start the Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
