import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
import shap

# 1. Generate synthetic dataset
def generate_synthetic_data(n_samples=10000, random_state=42):
    np.random.seed(random_state)

    data = pd.DataFrame({
        "household_size": np.random.randint(1, 8, size=n_samples),
        "garden_size": np.random.choice(["none", "small", "medium", "large"], size=n_samples),
        "has_dishwasher": np.random.choice([0, 1], size=n_samples),
        "washing_frequency": np.random.randint(1, 14, size=n_samples),
        "water_pressure": np.random.uniform(1.0, 5.0, size=n_samples)
    })

    # Targets (random but correlated for variety)
    y = pd.DataFrame({
        "shower_shorter": (data["household_size"] > 4).astype(int),
        "garden_optimize": (data["garden_size"].isin(["medium", "large"])).astype(int),
        "tap_off_brushing": np.random.choice([0, 1], size=n_samples),
        "use_dishwasher": data["has_dishwasher"],
        "fix_leaks": (data["water_pressure"] > 4).astype(int),
        "optimize_washer": (data["washing_frequency"] > 7).astype(int)
    })

    return data, y


# 2. Build pipeline
def build_and_train_pipeline(X_train, y_train):
    numeric_features = ["household_size", "washing_frequency", "water_pressure"]
    categorical_features = ["garden_size", "has_dishwasher"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )))
    ])

    clf.fit(X_train, y_train)

    # Return all necessary components for later use
    return clf, numeric_features, categorical_features, preprocessor


# 3. Train, evaluate, save, and generate SHAP explainers
def train_and_save():
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data()

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training pipeline on synthetic data...")
    pipeline, numeric_features, categorical_features, preproc = build_and_train_pipeline(X_train, y_train)

    print("Done training. Evaluating on test set...\n")
    y_pred = pipeline.predict(X_test)

    print("=== Classification reports (per target) ===\n")
    for i, col in enumerate(y.columns):
        print(f"--- {col} ---")
        print(classification_report(y_test[col], y_pred[:, i]))
        print()

    # Save model
    joblib.dump(pipeline, "models/pipeline.joblib")
    print("Saved pipeline to models/pipeline.joblib")

    # Generate SHAP explainers
    print("Generating SHAP explainers (this may take a few seconds)...")

    # Get feature names
    num_feat = preproc.named_transformers_["num"].get_feature_names_out(numeric_features)
    cat_feat = preproc.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_feat, cat_feat])

    # Use one of the classifiers inside the multioutput
    model_for_shap = pipeline.named_steps["classifier"].estimators_[0]
    explainer = shap.TreeExplainer(model_for_shap)

    # Transform test data for SHAP
    X_transformed = preproc.transform(X_test)
    shap_values = explainer.shap_values(X_transformed)

    joblib.dump({
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_names": feature_names
    }, "models/shap_explainer.joblib")
    print("Saved SHAP explainer to models/shap_explainer.joblib")


if __name__ == "__main__":
    train_and_save()
