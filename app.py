import streamlit as st
import time
import random
import pandas as pd
import cv2
import threading
import queue
from vision import MultimodalSensor
import warnings

# Silence the Python 3.14 Pydantic version warnings
warnings.filterwarnings("ignore", category=UserWarning)

from simulation import NetworkEnv
from orchestrator import FogOrchestrator

# ---------- BACKGROUND WORKER ----------
# We define this OUTSIDE the loop so it can run independently
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
    page_title="Vehicular Fog Intelligence Console",
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

# ---------- HEADER ----------
st.title("🚗 Vehicular Fog Intelligence Console")
status_banner = st.empty() 

left, right = st.columns([2.2, 1])

# ---------- NETWORK PANEL ----------
with left:
    st.subheader("📡 Live Network Twin")

    st.session_state.env.step(st.session_state.camera_distance)
    snapshot = st.session_state.env.state()

    latency_placeholder = st.empty()
    latency_placeholder.metric(
        label="Observed Network Latency",
        value=f"{snapshot.get('real_latency_ms', 0)} ms"
    )

    node_cols = st.columns(len(snapshot["fog_nodes"]))
    node_placeholders = []

    for idx, node in enumerate(snapshot["fog_nodes"]):
        with node_cols[idx]:
            st.info(node["id"])
            stats_box = st.empty() 
            node_placeholders.append(stats_box)
            
            with stats_box.container():
                st.write(f"CPU Load → {node['cpu']}%")
                st.write(f"Temperature → {node['temp']} °C")
                if "health_status" in node:
                    st.write(f"Health → **{node['health_status']}**")
                st.caption(f"Mode → {node['mode']}")

    vehicle_placeholder = st.empty()
    vehicle_df = pd.DataFrame(snapshot["vehicles"])
    vehicle_placeholder.dataframe(vehicle_df, width="stretch", hide_index=True)

# ---------- COMMAND PANEL ----------
with right:
    st.subheader("🤖 Command Center")

    offline_mode = st.checkbox("🛑 Simulate Network Blackout")
    
    if offline_mode:
        status_banner.error("🔴 **SYSTEM OFFLINE:** Cloud AI unreachable. Relying on Local OBU ML Reflexes.")
    else:
        ai_status = "Generative AI (Phi-3)" if st.session_state.brain.use_ai else "Heuristics"
        status_banner.success(f"🟢 **SYSTEM ONLINE:** Decision Engine → {ai_status}")

    auto_mode = st.checkbox("Enable Autonomous Optimization")

    def push_log(message):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.insert(0, f"[{timestamp}] {message}")

    if auto_mode:
        st.warning("Autonomous cycle active")
        intents_pool = ["Prioritize Speed", "Thermal Protection", "Energy Saving", "Traffic Load Balance"]
        sensed_intent = random.choice(intents_pool)
        st.write(f"Detected Intent → **{sensed_intent}**")

        with st.spinner("AI evaluating system state..."):
            if offline_mode: st.session_state.brain.use_ai = False
            outcome = st.session_state.brain.decide(snapshot, sensed_intent)
            if offline_mode: st.session_state.brain.use_ai = True

        st.json(outcome)
        push_log(f"AUTO → {outcome.get('reasoning')}")
        time.sleep(1.8)
        st.rerun()

    else:
        user_intent = st.text_input("Provide Intent", value="Prioritize Speed")
        trigger = st.button("Execute Optimization")

        if trigger:
            with st.spinner("Computing orchestration decision..."):
                if offline_mode: st.session_state.brain.use_ai = False
                outcome = st.session_state.brain.decide(snapshot, user_intent)
                if offline_mode: st.session_state.brain.use_ai = True
            st.json(outcome)
            push_log(f"MANUAL → {outcome.get('reasoning')}")

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
    st.subheader("👁️ Live Dashcam (YOLO Object Detection)")
    enable_camera = st.checkbox("Turn on Dashcam")

    if enable_camera:
        if "vision_ai" not in st.session_state:
            with st.spinner("Waking up YOLO AI Model..."):
                st.session_state.vision_ai = MultimodalSensor()
                
        frame_window = st.image([])
        scene_text = st.empty()
        
        # CHANGED: Using index 0 as per your test results
        cap = cv2.VideoCapture(1)
        
        while enable_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            # 1. Check if the background thread finished
            if not st.session_state.ai_queue.empty():
                log_msg = st.session_state.ai_queue.get()
                push_log(log_msg)
                draw_decisions()
                st.session_state.is_thinking = False

            # 2. Process Vision
            processed_frame, is_emergency, scene, distance = st.session_state.vision_ai.process_frame(frame)
            st.session_state.camera_distance = distance
            st.session_state.env.step(distance)
            
            live_state = st.session_state.env.state()
            
            # 3. Apply Blackout Logic to Latency
            live_latency = 9999 if offline_mode else live_state.get('real_latency_ms', 0)
            latency_placeholder.metric(label="Observed Network Latency", value=f"{live_latency} ms")
            
            # 4. Update UI Placeholders
            for idx, node in enumerate(live_state["fog_nodes"]):
                with node_placeholders[idx].container():
                    st.write(f"CPU Load → {node['cpu']}%")
                    st.write(f"Temperature → {node['temp']} °C")
                    st.write(f"Health → **{node['health_status']}**")
                    st.caption(f"Mode → {node['mode']}")
                    
            vehicle_df = pd.DataFrame(live_state["vehicles"])
            vehicle_placeholder.dataframe(vehicle_df, width="stretch", hide_index=True)
            
            edge_0 = live_state["fog_nodes"][0]
            
            # 5. Trigger Non-Blocking Threaded AI
            if edge_0['health_status'] == 'CRITICAL' and st.session_state.last_alert != 'CRITICAL' and not st.session_state.is_thinking:
                st.session_state.last_alert = 'CRITICAL'
                st.session_state.is_thinking = True 
                
                # FIXED: Passing st.session_state.ai_queue as an argument
                thread = threading.Thread(
                    target=background_ai_task, 
                    args=(st.session_state.brain, live_state.copy(), offline_mode, st.session_state.ai_queue)
                )
                thread.start()
                
            elif edge_0['health_status'] == 'Stable':
                st.session_state.last_alert = 'Stable'

            # 6. HUD Drawing
            hud_cpu = f"Live CPU: {edge_0['cpu']}%"
            hud_temp = f"Live Temp: {edge_0['temp']} C"
            hud_health = f"Health: {edge_0['health_status']}"
            
            if st.session_state.is_thinking:
                cv2.putText(processed_frame, "AI THINKING...", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(processed_frame, f"Active Policy: {user_intent}", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(processed_frame, hud_cpu, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, hud_temp, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            color = (0, 0, 255) if edge_0['health_status'] == 'CRITICAL' else (0, 255, 0)
            cv2.putText(processed_frame, hud_health, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if is_emergency: scene_text.error(f"🚨 HAZARD: {scene}")
            else: scene_text.info(f"🛣️ {scene}")

            final_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(final_img, channels="RGB")
            
        cap.release()

if not auto_mode and not enable_camera:
    time.sleep(1)
    st.rerun()