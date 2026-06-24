import joblib
import pandas as pd
import shap

MODEL_DIR = "backend/app/ml/models"


class SHAPExplainer:

    model = joblib.load(f"{MODEL_DIR}/credit_model.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    explainer = shap.TreeExplainer(model)

    @staticmethod
    def explain(data: dict):
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        df = df.reindex(
            columns=SHAPExplainer.feature_columns,
            fill_value=0
        )

        shap_values = SHAPExplainer.explainer.shap_values(df)

        values = shap_values[0]

        explanations = []

        for feature, value in zip(SHAPExplainer.feature_columns, values):
            explanations.append(
                {
                    "feature": feature,
                    "impact": round(float(value), 4)
                }
            )

        explanations = sorted(
            explanations,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )[:8]

        return {
            "top_risk_factors": explanations,
            "model": "XGBoost + SHAP Explainability"
        }