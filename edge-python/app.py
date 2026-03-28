import streamlit as st
import time
import random
import pandas as pd
import cv2
import threading
import queue
import warnings
import requests
import json
import psutil

# Importing your custom logic
from vision import MultimodalSensor
from simulation import NetworkEnv
from orchestrator import FogOrchestrator

# ---------- CLOUD CONFIGURATION ----------
# This connects to your Java Spring Boot Backend (Phase 3)
CLOUD_URL = "http://localhost:8080/api/v1/cloud/sync"

# Silence the Python 3.14 Pydantic version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- THE CLOUD BRIDGE ----------
def sync_to_cloud(hazard_type, decision, cpu, temp):
    """
    Sends real-time telemetry from the car to the Java Backend.
    """
    payload = {
        "vehicleId": "2306048",           # Your Roll Number
        "cpuLoad": cpu,
        "temperature": temp,
        "healthStatus": "CRITICAL" if temp > 75 else "STABLE",
        "hazardDetected": hazard_type,
        "aiReasoning": decision
    }
    
    try:
        # Use a short timeout so the UI doesn't lag if the cloud is slow
        response = requests.post(CLOUD_URL, json=payload, timeout=1)
        if response.status_code == 200:
            return True
    except Exception:
        return False # Silent fail: Car keeps operating in Edge-Native mode

# ---------- BACKGROUND WORKER ----------
def background_ai_task(brain, state_data, offline, res_queue):
    """Handles heavy AI thinking without freezing the camera feed."""
    original_ai = brain.use_ai
    if offline:
        brain.use_ai = False
        
    outcome = brain.decide(state_data, "THERMAL OVERLOAD DETECTED")
    mode_tag = "OFFLINE" if offline else "ONLINE AI"
    
    # Restore brain state
    brain.use_ai = original_ai
    
    # Drop the result into the physical queue object
    res_queue.put(f"🚨 {mode_tag} REFLEX → {outcome.get('reasoning')}")

# ---------- INITIAL SETUP ----------
st.set_page_config(
    page_title="ENGO: Vehicular Fog Intelligence Console",
    layout="wide"
)

def bootstrap():
    if "camera_distance" not in st.session_state:
        st.session_state.camera_distance = 0
    if "env" not in st.session_state:
        st.session_state.env = NetworkEnv()
    if "brain" not in st.session_state:
        st.session_state.brain = FogOrchestrator(use_local_ai=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = "Stable"
    if "ai_queue" not in st.session_state:
        st.session_state.ai_queue = queue.Queue()
    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False

bootstrap()

# ---------- HEADER & UI ----------
st.title("🚗 ENGO: Vehicular Fog Intelligence Console")
status_banner = st.empty() 

left, right = st.columns([2.2, 1])

# ---------- NETWORK PANEL ----------
with left:
    st.subheader("📡 Live Network Twin")

    st.session_state.env.step(st.session_state.camera_distance)
    snapshot = st.session_state.env.state()

    latency_placeholder = st.empty()
    node_cols = st.columns(len(snapshot["fog_nodes"]))
    node_placeholders = []

    for idx, node in enumerate(snapshot["fog_nodes"]):
        with node_cols[idx]:
            st.info(node["id"])
            stats_box = st.empty() 
            node_placeholders.append(stats_box)

    vehicle_placeholder = st.empty()

# ---------- COMMAND PANEL ----------
with right:
    st.subheader("🤖 Command Center")
    offline_mode = st.checkbox("🛑 Simulate Network Blackout")
    auto_mode = st.checkbox("Enable Autonomous Optimization")
    
    # Cloud Status Indicator
    cloud_status = st.empty()

    def push_log(message):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.insert(0, f"[{timestamp}] {message}")

    st.markdown("---")
    st.markdown("### 📜 Recent Decisions")
    decision_box = st.empty() 
    
    def draw_decisions():
        with decision_box.container():
            for item in st.session_state.history[:6]:
                st.text(item)
    draw_decisions()

    # --- YOLO DASHCAM FEED ---
    st.markdown("---")
    st.subheader("👁️ Live Dashcam (YOLO + Cloud Sync)")
    enable_camera = st.checkbox("Turn on Dashcam")

    if enable_camera:
        if "vision_ai" not in st.session_state:
            with st.spinner("Waking up YOLO AI Model..."):
                st.session_state.vision_ai = MultimodalSensor()
                
        frame_window = st.image([])
        scene_text = st.empty()
        
        cap = cv2.VideoCapture(0) # Change to 1 if 0 is your internal webcam
        
        while enable_camera:
            ret, frame = cap.read()
            if not ret: break
                
            # 1. Background AI Task Check
            if not st.session_state.ai_queue.empty():
                log_msg = st.session_state.ai_queue.get()
                push_log(log_msg)
                draw_decisions()
                st.session_state.is_thinking = False

            # 2. Process Computer Vision
            processed_frame, is_emergency, scene, distance = st.session_state.vision_ai.process_frame(frame)
            st.session_state.camera_distance = distance
            st.session_state.env.step(distance)
            live_state = st.session_state.env.state()
            edge_0 = live_state["fog_nodes"][0]

            # 3. BRIDGE TO CLOUD (Phase 3 Integration)
            # We send the YOLO detection and local node health to Java
            sync_success = sync_to_cloud(
                hazard_type=scene, 
                decision=st.session_state.last_alert, 
                cpu=edge_0['cpu'], 
                temp=edge_0['temp']
            )
            
            if sync_success:
                cloud_status.success("☁️ Cloud Sync: ACTIVE (Connected to Java)")
            else:
                cloud_status.warning("☁️ Cloud Sync: OFFLINE (Edge-Native)")

            # 4. Update UI Placeholders
            live_latency = 9999 if offline_mode else live_state.get('real_latency_ms', 0)
            latency_placeholder.metric(label="Observed Network Latency", value=f"{live_latency} ms")
            
            for idx, node in enumerate(live_state["fog_nodes"]):
                with node_placeholders[idx].container():
                    st.write(f"CPU → {node['cpu']}% | Temp → {node['temp']}°C")
                    st.write(f"Health → **{node['health_status']}**")
            
            vehicle_df = pd.DataFrame(live_state["vehicles"])
            vehicle_placeholder.dataframe(vehicle_df, width="stretch", hide_index=True)
            
            # 5. Threaded LLM Logic
            if edge_0['health_status'] == 'CRITICAL' and st.session_state.last_alert != 'CRITICAL' and not st.session_state.is_thinking:
                st.session_state.last_alert = 'CRITICAL'
                st.session_state.is_thinking = True 
                thread = threading.Thread(
                    target=background_ai_task, 
                    args=(st.session_state.brain, live_state.copy(), offline_mode, st.session_state.ai_queue)
                )
                thread.start()
            elif edge_0['health_status'] == 'Stable':
                st.session_state.last_alert = 'Stable'

            # 6. HUD Drawing
            color = (0, 0, 255) if edge_0['health_status'] == 'CRITICAL' else (0, 255, 0)
            cv2.putText(processed_frame, f"CPU: {edge_0['cpu']}%", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"TEMP: {edge_0['temp']}C", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(processed_frame, f"HEALTH: {edge_0['health_status']}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if st.session_state.is_thinking:
                cv2.putText(processed_frame, "AI ANALYZING...", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if is_emergency: scene_text.error(f"🚨 HAZARD: {scene}")
            else: scene_text.info(f"🛣️ {scene}")

            final_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(final_img, channels="RGB")
            
        cap.release()

# Auto-refresh if camera is off
if not enable_camera:
    time.sleep(1)
    st.rerun()