# 🚀 Custom LLM API – Optimized & Self-Hosted AI Model  

## 🔥 Overview  
This project is a **fully self-hosted, optimized AI inference API** that runs a custom **Large Language Model (LLM)** on personal hardware. 
With support for TVM, vLLM, XLA, ONNX, and DirectML, the API enables real hardware-aware optimization and efficient inference across NVIDIA, AMD, CPU, and TPU-compatible systems.
The model is highly efficient, making it a **cost-effective alternative to expensive commercial LLM APIs** like OpenAI's.

### **🔹 Why This is Useful**
Many businesses and developers rely on cloud-based LLM APIs like OpenAI, which can be expensive and require constant internet access.  
This project allows anyone to **run an LLM locally on their own hardware**, reducing costs and improving data privacy.

### **🔹 Example Use Case**
Imagine you're building a **customer support chatbot** for an **e-commerce website**. Instead of **paying per API call to OpenAI**, you can:  
✅ **Host this model on your own server**  
✅ **Integrate it into your chatbot application** via the API  
✅ **Customize the model's responses** to better fit your brand and support needs  

With **self-hosted LLMs**, businesses can have **full control over their AI models** while saving money and improving response times.

## 🏗 Tech Stack  

### **🔹 Supported LLMs**  
This project allows users to **select and deploy different LLMs** based on their needs:  
- **Lightweight Models (Fast, Low Resource Usage)**  
  - `GPT-2` → Small and efficient for basic text generation  
  - `Phi-1.5` → More advanced, but still runs on consumer hardware  

- **Mid-Size Models (Balanced Performance & Quality)**  
  - `Mistral-7B` → High-quality open-source model, better than GPT-3.5  
  - `Llama-2 7B` → Optimized for general AI tasks with strong efficiency  

- **Large Models (High Accuracy, Requires GPU/Cloud Deployment)**  
  - `Llama-2 13B` → More powerful, needs strong hardware  
  - `Mistral-7B Instruct` → Fine-tuned for chatbot-like applications  
  - `Mixtral-8x7B` → Multi-expert architecture for enhanced reasoning  

### **🔹 Optimization Techniques**
These methods **enhance model efficiency** and **reduce memory consumption**:  
- **TVM (Apache TVM)** → Compiles models for maximum performance on CPU/GPU  
- **vLLM** → Highly optimized inference engine for handling large models  
- **XLA (JIT Compilation)** → Speeds up matrix operations for deep learning  

### **🔹 Infrastructure & Deployment**
- **Flask** → Simple & scalable API framework  
- **Docker** → Makes deployment easy & portable  
- **Railway.app** → Cloud hosting for easy access  
- **ONNX Runtime** → Enables cross-platform execution & model optimizations  
- **DirectML (Windows)** → Supports AMD GPU acceleration on Windows  
- **CUDA (Linux/NVIDIA GPUs)** → Enables fast inference on dedicated GPUs  

## 🚀 Features  
✅ **Choose from multiple LLMs** – Supports models from **GPT-2 (lightweight) to Mistral-7B & Llama-2**  
✅ **Fast, optimized AI inference** – Uses **TVM, vLLM, and XLA** to speed up response times  
✅ **Runs on personal hardware or cloud GPUs** – Eliminates need for costly OpenAI API  
✅ **Supports model fine-tuning** – Customize responses for specific use cases  
✅ **Flexible deployment** – Works on **local machines, servers, or cloud GPUs**  
✅ **Lightweight & scalable API** – Built with **Flask** for easy integration into any application  
✅ **Cross-platform support** – Works on **Windows (DirectML), Linux (CUDA), and ONNX Runtime**  
✅ **Containerized with Docker** – Ensures easy deployment & portability  

---
## 💡 How This API Can Be Used  

This API is designed for **maximum flexibility**, so you can run it in different ways based on your needs:  

1️⃣ **Run Locally on Your Personal Hardware**  
   - If you want full control, just install the dependencies and run `server.py`.  
   - Works on **Windows (DirectML), Linux (CUDA), and CPU-only systems**.  

2️⃣ **Run Inside a Docker Container**  
   - If you need a **portable and reproducible** setup, Docker lets you run the API easily anywhere.  

3️⃣ **Deploy to a Cloud GPU (Railway, AWS, Google Cloud)**  
   - If you want to serve your LLM remotely, deploy it to **Railway.app or a cloud GPU** for public access.  

4️⃣ **Optimize Performance Based on Your Hardware**  
   - Uses **TVM, vLLM, and XLA** to maximize efficiency, depending on your hardware (CPU, AMD, NVIDIA).  

💡 **Choose the method that best fits your needs!**  
---

## 🛠 Installation & Running the API  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
cd YOUR_PROJECT
```

### **2️⃣ Set Up Virtual Environment & Install Dependencies**  
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate      # For Windows

pip install -r requirements.txt
```

### **3️⃣ Run the API Locally**
```bash
python server.py
```

### **4️⃣ Test the API**
Send a request to the model via **cURL**:
```bash
curl -X POST "http://127.0.0.1:5000/generate" -H "Content-Type: application/json" -d '{"text": "Tell me a fun fact."}'
```
✅ If everything is working, you should see a generated AI response!

---

## 🐳 Running with Docker  
To make deployment easier, you can **run the API in a Docker container**:  

### **1️⃣ Build the Docker Image**  
```bash
docker build -t my-ai-api .
```

### **2️⃣ Run the Container**  
```bash
docker run -p 5000:5000 my-ai-api
```
Now your **AI API is running inside a container!** 🚀

---

## 🌍 Deploying to Cloud (Railway.app)  
If you want to **host the API on the internet**, you can **deploy it to Railway.app**:  

### **1️⃣ Push to GitHub**  
```bash
git init
git add .
git commit -m "Initial commit for AI API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
git push -u origin main
```

### **2️⃣ Deploy on Railway.app**
1. Go to **[Railway.app](https://railway.app/)**  
2. Create a **New Project** → Select **"Deploy from GitHub"**  
3. Choose your repo and click **"Deploy"**  
4. Once deployed, you'll get a **public API URL** like:  
   ```
   https://your-ai-api.up.railway.app/generate
   ```

✅ **Now your AI API is live and can be accessed from anywhere!** 🎉

---

## ⚡ AI Optimization Breakdown  
Here’s how this project **makes LLM inference faster & more efficient**:  

| **Optimization** | **What It Does** | **Benefit** |
|----------------|----------------|----------------|
| **TVM (Apache TVM)** | Compiles the AI model into highly optimized machine code | **Speeds up inference on CPU/GPU** |
| **vLLM** | Efficient inference engine for large models | **Reduces memory usage & increases throughput** |
| **XLA (JIT Compilation)** | Just-in-time compilation for deep learning workloads | **Faster execution by optimizing tensor computations** |

✅ **These optimizations make the API significantly faster than standard LLM inference!** 🚀

---

## 🤖 Example Use Cases  
✅ **Custom AI Assistant** → Build an LLM-powered chatbot  
✅ **Automated Content Generation** → Create AI-generated blogs, summaries, or code snippets  
✅ **Research & Experimentation** → Optimize LLM inference for **faster, cheaper AI deployments**  


🚀 **Star the repo if you found this useful!** ⭐  
