# It centralizes the model, configuration and feature schema in one reusable class. Later,
# both single and  batch predictions will use this same prediction engine.
import json
from pathlib import Path
import mlflow
import mlflow.lightgbm
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

mlflow.set_tracking_uri("sqlite:///C:/mlflow_new/mlflow.db")


class FraudPredictor:
    def __init__(self):
        config_path = ARTIFACTS_DIR / "inference_config.json"
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        schema_path = ARTIFACTS_DIR / "inference_schema.json"
        with open(schema_path, "r", encoding="utf-8") as file:
            self.schema = json.load(file)

        model_uri = (
            f"models:/{self.config['model_name']}/"
            f"{self.config['model_version']}"
        )

        self.model = mlflow.lightgbm.load_model(model_uri)
        self.threshold = float(self.config["threshold"])

        # SHAP explainer for the native LightGBM model.
        self.explainer = shap.TreeExplainer(self.model)

        print("FraudPredictor initialized successfully")
        print(f"Model: {self.config['model_name']}")
        print(f"Version: {self.config['model_version']}")
        print(f"Features: {self.schema['feature_count']}")
        print(
            f"Categorical columns: "
            f"{len(self.schema['categorical_columns'])}"
        )
        print(f"Threshold: {self.threshold}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare transaction data for LightGBM inference."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Input transaction data is empty.")

        expected_features = self.schema["feature_names"]

        missing_features = [
            column
            for column in expected_features
            if column not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing {len(missing_features)} required features: "
                f"{missing_features[:10]}"
            )

        prepared = data[expected_features].copy()

        # Prepare categorical columns exactly as defined by the
        # inference schema.
        for column in self.schema["categorical_columns"]:
            categories = self.schema["categorical_categories"][column]

            if "__UNKNOWN__" not in categories:
                raise ValueError(
                    f"Missing __UNKNOWN__ category for {column}."
                )

            values = prepared[column].astype("string")

            if "__MISSING__" in categories:
                values = values.fillna("__MISSING__")

            values = values.where(
                values.isna() | values.isin(categories),
                "__UNKNOWN__"
            )

            prepared[column] = pd.Categorical(
                values,
                categories=categories
            )

        numeric_columns = [
            column
            for column in expected_features
            if column not in self.schema["categorical_columns"]
        ]

        for column in numeric_columns:
            try:
                prepared[column] = pd.to_numeric(
                    prepared[column],
                    errors="raise"
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Feature '{column}' contains invalid numeric values."
                ) from exc

        return prepared

    def predict(self, data: pd.DataFrame) -> dict:
        """Generate fraud probability and prediction for one transaction."""

        X = self.prepare_features(data)

        probability = float(self.model.predict_proba(X)[0, 1])
        prediction = int(probability >= self.threshold)

        risk_level = self._risk_level(probability)

        return {
            "fraud_probability": probability,
            "prediction": prediction,
            "is_fraud": bool(prediction),
            "risk_level": risk_level,
            "threshold": self.threshold,
        }

    def predict_batch(self, data: pd.DataFrame) -> list:
        """Generate fraud predictions for multiple transactions."""

        X = self.prepare_features(data)
        probabilities = self.model.predict_proba(X)[:, 1]

        results = []

        for probability in probabilities:
            probability = float(probability)
            prediction = int(probability >= self.threshold)

            results.append({
                "fraud_probability": probability,
                "prediction": prediction,
                "is_fraud": bool(prediction),
                "risk_level": self._risk_level(probability),
                "threshold": self.threshold,
            })

        return results

    def explain(self, data: pd.DataFrame, top_n: int = 5) -> dict:
        """
        Explain a single fraud prediction using SHAP.

        SHAP values represent how individual model features
        contributed to the model's fraud score.
        """

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        X = self.prepare_features(data)

        if len(X) != 1:
            raise ValueError(
                "Explainability requires exactly one transaction."
            )

        probability = float(self.model.predict(X)[0])
        prediction = int(probability >= self.threshold)
        risk_level = self._risk_level(probability)

        shap_values = self.explainer.shap_values(X)

        # LightGBM binary classifiers can return either:
        #   ndarray -> shape (1, n_features)
        # or
        #   list[ndarray] -> class-specific SHAP values.
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = shap_values[0]

        feature_names = X.columns.tolist()

        contributions = []

        for feature_name, shap_value in zip(
            feature_names,
            shap_values
        ):
            value = X.iloc[0][feature_name]

            contributions.append({
                "feature": feature_name,
                "value": self._format_feature_value(value),
                "shap_value": float(shap_value),
                "direction": (
                    "toward_fraud"
                    if shap_value > 0
                    else "away_from_fraud"
                ),
                "label": self._friendly_feature_label(feature_name),
            })

        # Strongest contributions in either direction.
        contributions.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        top_contributions = contributions[:top_n]

        positive = [
            item
            for item in contributions
            if item["shap_value"] > 0
        ]

        negative = [
            item
            for item in contributions
            if item["shap_value"] < 0
        ]


        # Build a simplified explanation for analysts.
        analyst_groups = {}

        displayable_features = {
            "TransactionAmt": "Transaction amount",
            "ProductCD": "Product category",
            "card4": "Card network",
            "card6": "Card type",
            "P_emaildomain": "Purchaser email domain",
            "R_emaildomain": "Recipient email domain",
            "DeviceType": "Device type",
            "DeviceInfo": "Device information",
            "id_30": "Operating system",
            "id_31": "Browser information",
            "id_33": "Screen resolution",
        }

        for item in contributions:
            label = item["label"]

            if label in {
                "Card identifier signal",
                "Identity verification status",
                "Identity signal",
                "Identity verification signal",
                "Identity match status",
                "Proxy / anonymity signal",
            }:
                group = "Identity and card signals"

            elif label in {
                "Anonymised model signal",
                "Anonymised identity signal",
            }:
                group = "Anonymised model signals"

            elif label == "Transaction time":
                group = "Transaction timing"

            else:
                group = label

            if group not in analyst_groups:
                analyst_groups[group] = {
                    "category": group,
                    "positive_contribution": 0.0,
                    "negative_contribution": 0.0,
                    "net_contribution": 0.0,
                    "important_factors": [],
                }

            shap_value = item["shap_value"]

            analyst_groups[group]["net_contribution"] += shap_value

            if shap_value > 0:
                analyst_groups[group]["positive_contribution"] += shap_value
            elif shap_value < 0:
                analyst_groups[group]["negative_contribution"] += shap_value

            # Only display values for features with an interpretable
            # business meaning. Do not expose anonymised feature codes.
            feature_name = item["feature"]

            if feature_name in displayable_features:
                analyst_groups[group]["important_factors"].append({
                    "feature": displayable_features[feature_name],
                    "value": item["value"],
                    "contribution": shap_value,
                    "direction": item["direction"],
                })

        analyst_explanation = []

        for group in analyst_groups.values():
            net = group["net_contribution"]

            if net > 0:
                direction = "toward_fraud"
                interpretation = "Increased the model's fraud score"
            elif net < 0:
                direction = "away_from_fraud"
                interpretation = "Reduced the model's fraud score"
            else:
                direction = "mixed"
                interpretation = "No net contribution to the fraud score"

            group["important_factors"].sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True
            )

            analyst_explanation.append({
                **group,
                "direction": direction,
                "interpretation": interpretation,
            })

        analyst_explanation.sort(
            key=lambda item: abs(item["net_contribution"]),
            reverse=True
        )
        explanation_note = (
            "SHAP contributions show how individual features influenced "
            "the model's fraud score. Positive contributions push the "
            "score toward fraud, while negative contributions push it "
            "away from fraud. These are model explanations, not proof "
            "of fraudulent behaviour."
        )

        return {
            "fraud_probability": probability,
            "prediction": prediction,
            "is_fraud": bool(prediction),
            "risk_level": risk_level,
            "threshold": self.threshold,
            "top_contributions": top_contributions,
            "top_positive_contributions": positive[:top_n],
            "top_negative_contributions": negative[:top_n],
            "explanation_note": explanation_note,
            "analyst_explanation": analyst_explanation,
        }

    def _risk_level(self, probability: float) -> str:
        """Convert fraud probability into the application's risk level."""

        if probability >= self.threshold:
            return "HIGH"

        if probability >= self.threshold / 2:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _format_feature_value(value):
        """Convert feature values into JSON-friendly display values."""

        if pd.isna(value):
            return None

        if isinstance(value, pd.Categorical):
            return str(value)

        return str(value)

    @staticmethod
    def _friendly_feature_label(feature_name: str) -> str:
        """
        Convert known features into analyst-friendly labels.

        Unknown/anonymised features are deliberately not given
        invented meanings.
        """

        labels = {
            "TransactionAmt": "Transaction amount",
            "TransactionDT": "Transaction time",
            "ProductCD": "Product category",
            "card1": "Card identifier signal",
            "card2": "Card identifier signal",
            "card3": "Card identifier signal",
            "card4": "Card network",
            "card5": "Card identifier signal",
            "card6": "Card type",
            "addr1": "Billing address signal",
            "addr2": "Billing region signal",
            "dist1": "Address distance signal",
            "dist2": "Address distance signal",
            "P_emaildomain": "Purchaser email domain",
            "R_emaildomain": "Recipient email domain",
            "DeviceType": "Device type",
            "DeviceInfo": "Device information",
            "id_12": "Identity verification status",
            "id_15": "Identity verification status",
            "id_16": "Identity verification status",
            "id_23": "Proxy / anonymity signal",
            "id_27": "Identity signal",
            "id_28": "Identity signal",
            "id_29": "Identity signal",
            "id_30": "Operating system",
            "id_31": "Browser information",
            "id_33": "Screen resolution",
            "id_34": "Identity match status",
            "id_35": "Identity verification signal",
            "id_36": "Identity verification signal",
            "id_37": "Identity verification signal",
            "id_38": "Identity verification signal",
        }

        if feature_name in labels:
            return labels[feature_name]

        if feature_name.startswith("V"):
            return "Anonymised model signal"

        if feature_name.startswith("C"):
            return "Anonymised model signal"

        if feature_name.startswith("D"):
            return "Anonymised model signal"

        if feature_name.startswith("M"):
            return "Anonymised model signal"

        if feature_name.startswith("id_"):
            return "Anonymised identity signal"

        return "Anonymised model signal"