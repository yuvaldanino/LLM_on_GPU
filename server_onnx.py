import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import numpy as np

# Import necessary libraries:
# - numpy for numerical operations
# - onnxruntime for running the ONNX model
# - transformers for loading the tokenizer
# - flask for creating the web API

# Load the tokenizer from the pre-trained GPT-2 model.
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load ONNX Model for Optimized Inference
onnx_path = "gpt2.onnx"
# Create an ONNX Runtime session for the model, specifying the execution provider.
# "DmlExecutionProvider" is used here for running on DirectML (typically for Windows with compatible hardware).
ort_session = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider"])

# Initialize Flask API
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.json
        # Extract the JSON data from the request.
        input_text = data.get("text", "")

        if not input_text:
            # If no input text is provided, return an error response.
            return jsonify({"error": "No input text provided"}), 400

        print("Received request:", input_text)  # Debugging print
        # Log the received input text for debugging purposes.

        # Convert the input text into token IDs using the tokenizer.
        # The token IDs are returned as a NumPy array with the "np" return type.
        input_ids = tokenizer(input_text, return_tensors="np")["input_ids"].astype(np.int64)

        # Ensure input is exactly 2D: (batch_size, sequence_length)
        # Adjust the shape of the input IDs to ensure it is 2D.
        if len(input_ids.shape) == 3:  # If it's (1,1,N), flatten it to (1,N)
            input_ids = input_ids.reshape(input_ids.shape[0], input_ids.shape[2])
        elif len(input_ids.shape) == 1:  # If it's just (N,), add batch dimension
            input_ids = input_ids.reshape(1, -1)

        print("✅ Converted text to input IDs:", input_ids)  # Debugging print
        print("✅ Input Shape:", input_ids.shape)  # Show the shape of the input

        # Run the ONNX model with the input IDs.
        outputs = ort_session.run(None, {"input_ids": input_ids})

        # Extract only the token IDs from ONNX output
        # Get the token IDs with the highest probability for each step in the output.
        token_ids = np.argmax(outputs[0], axis=-1)

        # Flatten array if needed
        # Ensure the token IDs array is 2D, flattening if necessary.
        if len(token_ids.shape) > 2:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)

        print("✅ Token IDs from ONNX Model:", token_ids)  # Debugging print
        # Log the token IDs obtained from the model for debugging purposes.

        # ✅ Decode output
        # Convert the token IDs back into human-readable text.
        generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)

        # Return the generated text as a JSON response.
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        # Handle any exceptions that occur during processing.
        print("❌ ERROR OCCURRED:", str(e))  # Print full error message
        # Log the error message for debugging purposes.
        return jsonify({"error": str(e)}), 500
        # Return an error response with the exception message.

# Start API
# If this script is run directly, start the Flask application.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
