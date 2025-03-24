import requests

def run_with_vllm(prompt):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["text"]
