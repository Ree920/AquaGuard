# continuous_simulation.py
import time
import joblib
from agent import AquaGuardAgent

def main():
    try:
        agent = joblib.load("agent_state.joblib")
        print("Loaded existing agent state.")
    except:
        agent = AquaGuardAgent()
        print("Initialized new agent.")

    for i in range(1000):
        sample, recs = agent.simulate_and_recommend()
        feedback = agent.simulate_feedback(sample, recs)
        agent.receive_feedback(sample, recs, feedback)

        if i % 10 == 0:
            joblib.dump(agent, "agent_state.joblib")
            print(f"Saved agent state at iteration {i}")

        print(f"Iteration {i+1} done.")
        time.sleep(2)

if __name__ == "__main__":
    main()