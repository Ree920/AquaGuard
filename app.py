# app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import joblib
import numpy as np

from agent import AquaGuardAgent

# Load or initialize agent
@st.cache_resource
def load_agent():
    try:
        return joblib.load("agent_state.joblib")
    except:
        return AquaGuardAgent()

agent = load_agent()

st.title("AquaGuard Home Agent - Continuous Simulation")

# Auto refresh every 5 seconds, max 1000 times
count = st_autorefresh(interval=5000, limit=1000, key="refresh")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Simulate one cycle of perception -> reasoning -> action -> learning
sample, recs = agent.simulate_and_recommend()

# Simulate user feedback (simple heuristic or random)
feedback = agent.simulate_feedback(sample, recs)

# Agent learns from feedback
agent.receive_feedback(sample, recs, feedback)

# Save updated agent state to disk for standalone script and next app load
joblib.dump(agent, "agent_state.joblib")

# Record history for display
st.session_state.history.append({
    "sample": sample,
    "recommendations": recs,
    "feedback": feedback
})

# Show latest cycle info
st.subheader(f"Cycle #{len(st.session_state.history)}")

st.markdown("### Simulated Input:")
st.json(sample)

st.markdown("### Recommendations:")
for r in recs:
    st.write(f"- {r}")

st.markdown("### User Feedback:")
st.json(feedback)

# Optionally show full history or stats
if st.checkbox("Show full history"):
    st.write(st.session_state.history)