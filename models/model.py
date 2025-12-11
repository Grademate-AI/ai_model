import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from civismart_ai.llm_client import call_llm_extract_features
from civismart_ai.utils import assign_urgency

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_model.pkl")

FEATURE_COLS = [
    "danger_score",
    "health_risk",
    "environment_damage",
    "infrastructure_damage",
    "fire_risk",
    "flooding_indicator",
    "urgency_keywords_score",
    "human_harm_risk"
]

class CiviSmartAI:
    def __init__(self, dataset_csv=None):
        self.dataset_csv = dataset_csv
        self.model = None
        self.le = None

        # Load model + encoder if available
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.le = joblib.load(ENCODER_PATH)

    def train(self):
        """Train the model from dataset"""
        if self.dataset_csv is None:
            raise ValueError("Dataset CSV required for training.")

        df = pd.read_csv(self.dataset_csv)

        # Generate labels if missing
        if "urgency_label" not in df.columns:
            df["urgency_label"] = df.apply(assign_urgency, axis=1)

        features_df = df[FEATURE_COLS]
        labels = df["urgency_label"]

        # Encode labels
        self.le = LabelEncoder()
        labels_enc = self.le.fit_transform(labels)

        # Train logistic regression
        self.model = LogisticRegression(max_iter=2000, solver='lbfgs')
        self.model.fit(features_df, labels_enc)

        # Save with joblib
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.le, ENCODER_PATH)

        print("✅ Model trained & saved successfully")

    def predict(self, report_text, use_openai=False):
        """Predict urgency for a new report"""
        if self.model is None or self.le is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Extract features using LLM
        features = call_llm_extract_features(report_text, use_openai=use_openai)

        # Convert feature dict → DataFrame with proper column names
        features_df = pd.DataFrame([features], columns=FEATURE_COLS)

        pred_encoded = self.model.predict(features_df)[0]
        urgency_label = self.le.inverse_transform([pred_encoded])[0]

        return {
            "urgency": urgency_label,
            "features": features
        }

    def retrain_with_new_data(self, new_data_csv):
        """Retrain model with additional verified data"""
        if self.dataset_csv is None:
            raise ValueError("Original dataset CSV required.")

        df_old = pd.read_csv(self.dataset_csv)
        df_new = pd.read_csv(new_data_csv)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)

        # Rewrite updated dataset
        df_combined.to_csv(self.dataset_csv, index=False)

        print(f"📌 Dataset expanded → {len(df_combined)} rows")
        self.train()
