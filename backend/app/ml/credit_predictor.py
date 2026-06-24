import joblib
import pandas as pd

MODEL_DIR = "backend/app/ml/models"


class CreditPredictor:

    model = joblib.load(
        f"{MODEL_DIR}/credit_model.pkl"
    )

    feature_columns = joblib.load(
        f"{MODEL_DIR}/feature_columns.pkl"
    )

    @staticmethod
    def predict(data: dict):

        df = pd.DataFrame([data])

        df = pd.get_dummies(df)

        df = df.reindex(
            columns=CreditPredictor.feature_columns,
            fill_value=0
        )

        probability = float(
            CreditPredictor.model.predict_proba(df)[0][1]
        )

        prediction = int(
            CreditPredictor.model.predict(df)[0]
        )

        credit_score = round(
            (1 - probability) * 100
        )

        risk_level = (
            "LOW"
            if probability < 0.30
            else "MODERATE"
            if probability < 0.60
            else "HIGH"
        )

        return {
            "credit_score": credit_score,
            "default_probability": round(
                probability,
                3
            ),
            "risk_prediction": prediction,
            "risk_level": risk_level
        }