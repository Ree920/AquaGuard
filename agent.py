# agent.py
import numpy as np

class AquaGuardAgent:
    def __init__(self):
        # Initialize model, history, thresholds etc.
        self.model = None  # placeholder for ML model if used
        self.history = []

    def simulate_and_recommend(self):
        # Simulate perception: generate random or rule-based user data
        sample = self._simulate_user_input()

        # Reasoning: generate recommendations based on sample
        recs = self._generate_recommendations(sample)

        return sample, recs

    def simulate_feedback(self, sample, recs):
        # Simulate user feedback: which recommendations followed
        feedback = {}
        for r in recs:
            # Randomly assume 50% acceptance for demo
            feedback[r] = np.random.choice([True, False])
        return feedback

    def receive_feedback(self, sample, recs, feedback):
        # Update internal state or model with feedback (learning step)
        self.history.append({
            "sample": sample,
            "recommendations": recs,
            "feedback": feedback
        })
        # Placeholder: could update ML model incrementally here

    def _simulate_user_input(self):
        # Simulated user inputs
        return {
            "shower_duration_min": np.random.randint(1, 15),
            "garden_watering_freq_per_week": np.random.randint(0, 7),
            "tap_off_while_brushing": np.random.choice([True, False]),
            "dishwasher_use": np.random.choice([True, False]),
        }

    def _generate_recommendations(self, sample):
        recs = []
        if sample["shower_duration_min"] > 10:
            recs.append("Consider reducing shower time to save water.")
        if sample["garden_watering_freq_per_week"] > 4:
            recs.append("Water your garden early morning or late evening.")
        if not sample["tap_off_while_brushing"]:
            recs.append("Turn off tap while brushing your teeth.")
        if not sample["dishwasher_use"]:
            recs.append("Using dishwasher can save water over hand washing.")
        return recs