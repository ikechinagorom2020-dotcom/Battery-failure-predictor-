import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from .model import BatteryFailureNet


class Predictor:
    """
    Loads the trained model + every preprocessing artifact (imputers,
    scaler, label encoders, feature config) and reproduces the exact
    same pipeline used in the training notebook (Cells 4-6) at
    inference time.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        with open(os.path.join(artifacts_dir, "feature_config.json")) as f:
            self.config = json.load(f)

        self.num_cols = self.config["num_cols"]
        self.cat_cols = self.config["cat_cols"]
        self.skewed_cols = self.config["skewed_cols"]
        self.threshold = self.config["best_threshold"]
        self.hidden_sizes = self.config.get("hidden_sizes", [128, 64, 32])
        self.dropout = self.config.get("dropout", 0.3)

        self.num_imputer = joblib.load(os.path.join(artifacts_dir, "num_imputer.pkl"))
        self.cat_imputer = joblib.load(os.path.join(artifacts_dir, "cat_imputer.pkl"))
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(artifacts_dir, "label_encoders.pkl"))

        input_dim = len(self.num_cols) + len(self.cat_cols)
        self.model = BatteryFailureNet(
            input_dim=input_dim, hidden_sizes=self.hidden_sizes, dropout=self.dropout
        )
        state_dict = torch.load(
            os.path.join(artifacts_dir, "best_model.pt"), map_location="cpu"
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _preprocess(self, raw: dict) -> np.ndarray:
        df = pd.DataFrame([raw])

        # Ensure every expected column is present (missing -> NaN -> imputed)
        for col in self.num_cols + self.cat_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Step 1: impute (same strategy as training: median / most_frequent)
        df[self.num_cols] = self.num_imputer.transform(df[self.num_cols])
        df[self.cat_cols] = self.cat_imputer.transform(df[self.cat_cols])

        # Step 2: encode categoricals (unseen categories fall back to the
        # first known class rather than raising, so the API stays robust)
        for col in self.cat_cols:
            le = self.label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

        # Step 3: log1p the same skewed columns identified during training
        for col in self.skewed_cols:
            df[col] = np.log1p(df[col].clip(lower=0))

        # Step 4: scale numeric columns with the fitted StandardScaler
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        X = df[self.num_cols + self.cat_cols].values.astype(np.float32)
        return X

    def predict(self, raw: dict) -> dict:
        X = self._preprocess(raw)
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            prob = torch.sigmoid(logits).item()
        prediction = int(prob >= self.threshold)
        return {
            "failure_probability": round(prob, 4),
            "prediction": prediction,
            "label": "Failure" if prediction == 1 else "No Failure",
            "threshold_used": self.threshold,
        }
