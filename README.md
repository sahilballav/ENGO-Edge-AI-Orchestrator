# 🚗 Vehicular Fog Intelligence Console (VFC)
### An Edge-Native Orchestration & Real-Time MLOps Framework

## 📖 Project Overview
The **Vehicular Fog Intelligence Console** is a high-performance prototype for autonomous vehicle orchestration. It solves the critical challenge of maintaining system intelligence during network blackouts by utilizing **Edge Computing** and a **Hybrid Decision Engine**.

The system senses physical hazards via a **YOLOv8 dashcam**, simulates real-time compute stress on edge nodes, and utilizes an asynchronous **MLOps pipeline** to retrain its health-prediction models on the fly.

---

## 🏗️ System Architecture
The architecture is built on a **Decoupled Microservice Pattern**, separating live inference from background model training to ensure zero-latency camera feeds.

### 1. Perception Layer (The Eyes)
* **YOLOv8 Vision Pipeline**: Real-time object detection and Z-axis distance estimation.
* **Sensor Fusion**: Translates optical proximity into virtual CPU/Thermal load spikes.

### 2. Hybrid Orchestration (The Brain)
* **Fast Reflex (ML)**: A Naive Bayes classifier trained to catch "Safety-Critical" keywords in **< 1ms**.
* **Cognitive Reasoning (LLM)**: A local **Phi-3 model** (via Ollama) for complex system optimization.
* **Asynchronous Concurrency**: Python Threading and Queues manage heavy AI tasks without blocking the UI thread.

### 3. Continuous MLOps (The Learning)
* **Telemetry Database**: Persistent logging of live physical states (CPU, Temp, Status) to a CSV-based "Data Lake."
* **Background Trainer**: A standalone microservice (`live_trainer.py`) that monitors data growth and hot-swaps the Random Forest `.pkl` model.

---

## 🛠️ Tech Stack
* **Language**: Python 3.14
* **UI/Dashboard**: Streamlit (Reactive Framework)
* **Vision**: OpenCV, Ultralytics YOLOv8
* **Orchestration**: LangChain, Ollama (Phi-3)
* **ML/Data**: Scikit-Learn (Random Forest, Naive Bayes), Pandas, NumPy, Joblib
* **Concurrency**: Threading, Queueing

---

## 🚀 Installation & Usage

### 1. Clone & Setup
```bash
git clone [https://github.com/YOUR_USERNAME/VFC-Intelligence-Console.git](https://github.com/sahilballav/VFC-Intelligence-Console.git)
cd VFC-Intelligence-Console
pip install -r requirements.txt
