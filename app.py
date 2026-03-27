import streamlit as st
import time
import random
import pandas as pd

from simulation import NetworkEnv
from orchestrator import FogOrchestrator


# ---------- INITIAL SETUP ----------
st.set_page_config(
    page_title="Vehicular Fog Intelligence Console",
    layout="wide"
)

def bootstrap():
    if "env" not in st.session_state:
        st.session_state.env = NetworkEnv()

    if "brain" not in st.session_state:
        st.session_state.brain = FogOrchestrator(use_local_ai=True)

    if "history" not in st.session_state:
        st.session_state.history = []

bootstrap()


# ---------- HEADER ----------
st.title("🚗 Vehicular Fog Intelligence Console")
st.markdown(
    f"**Decision Engine:** {'Local AI Model' if st.session_state.brain.use_ai else 'Heuristic Mode'}"
)

left, right = st.columns([2.2, 1])


# ---------- NETWORK PANEL ----------
with left:

    st.subheader("📡 Live Network Twin")

    st.session_state.env.step()
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
            st.caption(f"Mode → {node['mode']}")

    vehicle_df = pd.DataFrame(snapshot["vehicles"])
    st.dataframe(vehicle_df, use_container_width=True, hide_index=True)


# ---------- COMMAND PANEL ----------
with right:

    st.subheader("🤖 Command Center")

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
            outcome = st.session_state.brain.decide(snapshot, sensed_intent)

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
                outcome = st.session_state.brain.decide(snapshot, user_intent)

            elapsed = round(time.time() - t0, 2)

            st.success(f"Decision computed in {elapsed}s")
            st.json(outcome)

            push_log(f"MANUAL → {outcome.get('reasoning')}")

    st.markdown("---")
    st.markdown("### 📜 Recent Decisions")

    for item in st.session_state.history[:6]:
        st.text(item)


# ---------- PASSIVE REFRESH ----------
if not auto_mode:
    time.sleep(1)
    st.rerun()