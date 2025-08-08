import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class AquaGuardAgent:
    def __init__(self):
        self.feature_names = [
            "shower_duration_min",
            "garden_watering_freq_per_week",
            "tap_off_while_brushing",
            "dishwasher_use"
        ]
        self.actions = [
            "reduce_shower_time",
            "water_garden_early_morning",
            "turn_off_tap_while_brushing",
            "use_dishwasher"
        ]

        # Preprocessing pipeline: scaler for numeric + bool to int conversion
        self.scaler = StandardScaler()

        # Initialize model with multi-output binary classifiers (one per action)
        # We'll use one SGDClassifier per action wrapped in a dict for simplicity
        self.models = {}
        for action in self.actions:
            model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
            # We will call partial_fit with classes=[0,1]
            self.models[action] = model

        # Keep track if model was ever trained
        self.trained = {action: False for action in self.actions}

        # History log
        self.history = []

    def simulate_and_recommend(self):
        sample = self._simulate_user_input()
        X = self._prepare_features(sample)

        # Predict probability of acceptance for each action
        recs = []
        for action in self.actions:
            model = self.models[action]
            if self.trained[action]:
                prob = model.predict_proba(X)[0, 1]  # probability of class 1 (accept)
                # Recommend if predicted prob > 0.5 (tunable)
                if prob > 0.5:
                    recs.append(self._action_to_text(action))
            else:
                # If model not trained yet, fallback to rules
                recs.extend(self._rule_based_recommendations(sample))
                break  # no point checking others

        if not recs:
            # If none passed threshold, fallback to rules to keep agent active
            recs = self._rule_based_recommendations(sample)

        return sample, recs

    def simulate_feedback(self, sample, recs):
        # Simulate user feedback: user accepts 70% of recommended actions if rule triggered, else 30%
        feedback = {}
        for action_text in recs:
            # Map text back to action key
            action = self._text_to_action(action_text)
            # Simple heuristic for acceptance probability
            accept_prob = 0.7 if self._rule_triggered(sample, action) else 0.3
            feedback[action] = np.random.rand() < accept_prob
        return feedback

    def receive_feedback(self, sample, recs, feedback):
        # Use feedback to train models incrementally (partial_fit)
        X = self._prepare_features(sample)
        for action in self.actions:
            y = np.array([int(feedback.get(action, False))])  # 1 if accepted else 0
            model = self.models[action]
            if not self.trained[action]:
                # partial_fit needs classes specified on first call
                model.partial_fit(X, y, classes=[0, 1])
                self.trained[action] = True
            else:
                model.partial_fit(X, y)

        # Log for history
        self.history.append({
            "sample": sample,
            "recommendations": recs,
            "feedback": feedback
        })

    def _prepare_features(self, sample):
        # Convert input dict to numeric array, encode bool as int
        X = np.array([[
            sample["shower_duration_min"],
            sample["garden_watering_freq_per_week"],
            int(sample["tap_off_while_brushing"]),
            int(sample["dishwasher_use"])
        ]])

        # Scale features
        if not hasattr(self, "scaler_fitted") or not self.scaler_fitted:
            self.scaler.fit(X)
            self.scaler_fitted = True

        X_scaled = self.scaler.transform(X)
        return X_scaled

    def _simulate_user_input(self):
        return {
            "shower_duration_min": np.random.randint(1, 20),
            "garden_watering_freq_per_week": np.random.randint(0, 7),
            "tap_off_while_brushing": np.random.choice([True, False]),
            "dishwasher_use": np.random.choice([True, False]),
        }

    def _rule_based_recommendations(self, sample):
        recs = []
        if sample["shower_duration_min"] > 10:
            recs.append(self._action_to_text("reduce_shower_time"))
        if sample["garden_watering_freq_per_week"] > 4:
            recs.append(self._action_to_text("water_garden_early_morning"))
        if not sample["tap_off_while_brushing"]:
            recs.append(self._action_to_text("turn_off_tap_while_brushing"))
        if not sample["dishwasher_use"]:
            recs.append(self._action_to_text("use_dishwasher"))
        return recs

    def _rule_triggered(self, sample, action):
        # Check if rule triggers the action for the given sample
        if action == "reduce_shower_time":
            return sample["shower_duration_min"] > 10
        if action == "water_garden_early_morning":
            return sample["garden_watering_freq_per_week"] > 4
        if action == "turn_off_tap_while_brushing":
            return not sample["tap_off_while_brushing"]
        if action == "use_dishwasher":
            return not sample["dishwasher_use"]
        return False

    def _action_to_text(self, action):
        texts = {
            "reduce_shower_time": "Consider reducing shower time to save water.",
            "water_garden_early_morning": "Water your garden early morning or late evening.",
            "turn_off_tap_while_brushing": "Turn off tap while brushing your teeth.",
            "use_dishwasher": "Using dishwasher can save water over hand washing."
        }
        return texts.get(action, action)

    def _text_to_action(self, text):
        mapping = {v: k for k, v in {
            "reduce_shower_time": "Consider reducing shower time to save water.",
            "water_garden_early_morning": "Water your garden early morning or late evening.",
            "turn_off_tap_while_brushing": "Turn off tap while brushing your teeth.",
            "use_dishwasher": "Using dishwasher can save water over hand washing."
        }.items()}
        return mapping.get(text, text)