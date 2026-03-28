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

### 3. Cloud Orchestration Layer (The Headquarters)
* **Spring Boot Orchestrator**: A centralized Java-based backend that manages fleet telemetry and provides a RESTful sync interface.
* **H2 Database (SQL)**: Stores long-term telemetry records for academic audit and historical trend analysis.
* **Validation Logic**: Ensures that data coming from the edge (vehicle) matches the expected schema before being persisted.

### 4. Fog & Messaging Layer (The Infrastructure)
* **Apache Kafka (Event Streaming)**: Acts as a "Message Bus," allowing vehicle data to be streamed to other services without slowing down the main backend.
* **Redis (Live Caching)**: Provides sub-millisecond access to the "Current Status" of the car via RAM.
* **Zookeeper**: Manages the Kafka broker cluster state.

---

## 🔬 Research & Analytical Lab (Academic Module)
While the core console runs on a production-grade Random Forest model, this repository includes a **Research Lab** (Jupyter Notebooks) used to perform deep statistical analysis on the generated telemetry data.

### 1. **Linear Regression: Predictive Hardware Modeling**
* **Goal**: Model the correlation between CPU Load and Node Temperature.
* **Insight**: Used to calculate a "Line of Best Fit" to predict when a node will reach a thermal threshold before it actually happens ($y = mx + b$).

### 2. **K-Means Clustering: Unsupervised Behavioral Profiling**
* **Goal**: Group node states without using pre-defined labels.
* **Insight**: Discovers hidden patterns in hardware stress, identifying "Natural Clusters" of behavior.

### 3. **Isolation Forest: Anomaly & Security Detection**
* **Goal**: Identify "Outliers" in the telemetry stream.
* **Insight**: Acts as a cybersecurity layer to detect sensor spoofing or hardware tampering.

---

## 🛠️ Tech Stack
* **AI & Vision**: Python 3.x, OpenCV, Ultralytics YOLOv8, Scikit-Learn.
* **Cloud Backend**: Java 17, Spring Boot 3.x, Spring Data JPA, Maven.
* **Infrastructure**: Docker, Apache Kafka, Zookeeper, Redis.
* **Persistence**: H2 (In-Memory SQL), Pandas (CSV Data Lake).
* **UI**: Streamlit (Reactive Framework).

---

## 🚀 Installation & Execution

### 1. Clone & Setup
```bash
git clone [https://github.com/sahilballav/VFC-Intelligence-Console.git](https://github.com/sahilballav/VFC-Intelligence-Console.git)
cd VFC-Intelligence-Console
pip install -r requirements.txt
