import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/german_credit_data.csv"
MODEL_DIR = "backend/app/ml/models"

os.makedirs(MODEL_DIR, exist_ok=True)


def create_synthetic_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    risk_score = 0

    risk_score += (df["Credit amount"] > 8000).astype(int)
    risk_score += (df["Duration"] > 36).astype(int)
    risk_score += (df["Age"] < 25).astype(int)
    risk_score += (df["Housing"] == "rent").astype(int)
    risk_score += (df["Saving accounts"].fillna("unknown") == "little").astype(int)
    risk_score += (df["Checking account"].fillna("unknown") == "little").astype(int)

    df["Risk"] = (risk_score >= 2).astype(int)

    return df


def train_credit_model():
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = create_synthetic_risk_label(df)

    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X = pd.get_dummies(X, drop_first=True)

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"{MODEL_DIR}/credit_model.pkl")
    joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")

    print("XGBoost credit risk model saved successfully.")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train_credit_model()