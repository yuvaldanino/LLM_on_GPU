from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

# ✅ Use vLLM for optimized inference
model_name = "gpt2"  # You can swap this later for a better model
llm = LLM(model=model_name)

# ✅ Initialize Flask API
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        # Step 1: Get input text from request
        data = request.json
        input_text = data.get("text", "")

        if not input_text:
            return jsonify({"error": "No input text provided"}), 400

        # Step 2: Use vLLM for faster inference
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        outputs = llm.generate([input_text], sampling_params)

        # Step 3: Return the generated response
        generated_text = outputs[0].outputs[0].text

        return jsonify({"generated_text": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
