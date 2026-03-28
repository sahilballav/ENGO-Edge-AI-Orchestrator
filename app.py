import streamlit as st
import time
import random
import pandas as pd
import cv2
from vision import MultimodalSensor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from simulation import NetworkEnv
from orchestrator import FogOrchestrator


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

bootstrap()


# ---------- HEADER ----------
st.title("🚗 Vehicular Fog Intelligence Console")
# Create a dynamic placeholder that we will fill later based on the toggle
status_banner = st.empty() 

left, right = st.columns([2.2, 1])


# ---------- NETWORK PANEL ----------
with left:

    st.subheader("📡 Live Network Twin")

    # Feed the live camera data into the physics engine!
    st.session_state.env.step(st.session_state.camera_distance)
    snapshot = st.session_state.env.state()

    latency_value = snapshot.get("real_latency_ms", 0)

    st.metric(
        label="Observed Network Latency",
        value=f"{latency_value} ms",
        delta=None
    )

    node_cols = st.columns(len(snapshot["fog_nodes"]))

    for idx, node in enumerate(snapshot["fog_nodes"]):
        with node_cols[idx]:
            st.info(node["id"])
            st.write(f"CPU Load → {node['cpu']}%")
            st.write(f"Temperature → {node['temp']} °C")
            # This will now display the ML predictions from your simulation.py update!
            if "health_status" in node:
                st.write(f"Health → **{node['health_status']}**")
            st.caption(f"Mode → {node['mode']}")

    vehicle_df = pd.DataFrame(snapshot["vehicles"])
    # Fixed the Streamlit warning here
    st.dataframe(vehicle_df, width="stretch", hide_index=True)


# ---------- COMMAND PANEL ----------
with right:

    st.subheader("🤖 Command Center")

    # --- NEW: OFFLINE MODE TOGGLE ---
    offline_mode = st.checkbox("🛑 Simulate Network Blackout")
    
    if offline_mode:
        # Override latency to simulate a dead network
        snapshot["real_latency_ms"] = 9999
        # Dynamically inject the RED offline banner to the top of the page!
        status_banner.error("🔴 **SYSTEM OFFLINE:** Cloud AI unreachable. Relying on Local OBU ML Reflexes.")
    else:
        # Dynamically inject the GREEN online banner to the top of the page!
        ai_status = "Generative AI (Phi-3)" if st.session_state.brain.use_ai else "Heuristics"
        status_banner.success(f"🟢 **SYSTEM ONLINE:** Decision Engine → {ai_status}")

    # FIXED: Re-added the missing auto_mode checkbox here!
    auto_mode = st.checkbox("Enable Autonomous Optimization")

    def push_log(message):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.insert(0, f"[{timestamp}] {message}")

    if auto_mode:

        st.warning("Autonomous cycle active")

        intents_pool = [
            "Prioritize Speed",
            "Thermal Protection",
            "Energy Saving",
            "Traffic Load Balance"
        ]

        sensed_intent = random.choice(intents_pool)

        st.write(f"Detected Intent → **{sensed_intent}**")

        with st.spinner("AI evaluating system state..."):
            # Turn off AI to simulate network failure
            if offline_mode:
                st.session_state.brain.use_ai = False
                
            outcome = st.session_state.brain.decide(snapshot, sensed_intent)
            
            # Turn AI back on internally
            if offline_mode:
                st.session_state.brain.use_ai = True

        st.json(outcome)

        push_log(f"AUTO → {outcome.get('reasoning')}")

        time.sleep(1.8)
        st.rerun()

    else:

        user_intent = st.text_input(
            "Provide Intent",
            value="Prioritize Speed"
        )

        trigger = st.button("Execute Optimization")

        if trigger:

            t0 = time.time()

            with st.spinner("Computing orchestration decision..."):
                # Turn off AI to simulate network failure
                if offline_mode:
                    st.session_state.brain.use_ai = False

                outcome = st.session_state.brain.decide(snapshot, user_intent)

                # Turn AI back on internally
                if offline_mode:
                    st.session_state.brain.use_ai = True

            elapsed = round(time.time() - t0, 2)

            st.success(f"Decision computed in {elapsed}s")
            st.json(outcome)

            push_log(f"MANUAL → {outcome.get('reasoning')}")

    st.markdown("---")
    st.markdown("### 📜 Recent Decisions")

    for item in st.session_state.history[:6]:
        st.text(item)


# --- NEW: YOLO DASHCAM FEED ---
    st.markdown("---")
    st.subheader("👁️ Live Dashcam (YOLO Object Detection)")
    enable_camera = st.checkbox("Turn on Dashcam")

    if enable_camera:
        # Initialize the AI Vision pipeline in session state
        if "vision_ai" not in st.session_state:
            with st.spinner("Waking up YOLO AI Model..."):
                st.session_state.vision_ai = MultimodalSensor()
                
        # Create an empty box to display the video
        frame_window = st.image([])
        scene_text = st.empty()
        
        # Access the laptop webcam (0 is the default camera)
        cap = cv2.VideoCapture(0)
        
        while enable_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
                
            # 1. Get the frame and the physical distance of the object from vision.py
            processed_frame, is_emergency, scene, distance = st.session_state.vision_ai.process_frame(frame)
            
            # 2. FORCE the physics engine to update continuously inside this camera loop!
            st.session_state.env.step(distance)
            
            # 3. Grab the live physics data for Edge_0
            live_state = st.session_state.env.state()
            edge_0 = live_state["fog_nodes"][0]
            
            # 4. DRAW THE HEADS-UP DISPLAY (HUD) ON THE CAMERA FEED
            # We scale the distance down slightly so it doesn't get stuck at 100% too easily
            hud_cpu = f"Live CPU: {edge_0['cpu']}%"
            hud_temp = f"Live Temp: {edge_0['temp']} C"
            hud_health = f"Health: {edge_0['health_status']}"
            
            # Overlay the text using OpenCV (BGR colors: Green, Orange, Red)
            cv2.putText(processed_frame, hud_cpu, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, hud_temp, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Make the health text red if it is critical, otherwise green
            color = (0, 0, 255) if edge_0['health_status'] == 'CRITICAL' else (0, 255, 0)
            cv2.putText(processed_frame, hud_health, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show the text of what it sees in the Streamlit UI
            if is_emergency:
                scene_text.error(f"🚨 HAZARD: {scene}")
            else:
                scene_text.info(f"🛣️ {scene}")

            # Convert to RGB and display in Streamlit
            final_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(final_img, channels="RGB")
        cap.release()

# ---------- PASSIVE REFRESH ----------
if not auto_mode:
    time.sleep(1)
    st.rerun()