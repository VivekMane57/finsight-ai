from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.ml.credit_predictor import CreditPredictor
from backend.app.ml.shap_explainer import SHAPExplainer

router = APIRouter()


class CreditRequest(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Credit_amount: int
    Duration: int
    Purpose: str


def build_payload(data: CreditRequest) -> dict:
    return {
        "Age": data.Age,
        "Sex": data.Sex,
        "Job": data.Job,
        "Housing": data.Housing,
        "Saving accounts": data.Saving_accounts,
        "Checking account": data.Checking_account,
        "Credit amount": data.Credit_amount,
        "Duration": data.Duration,
        "Purpose": data.Purpose
    }


@router.post("/predict")
def predict_credit_risk(data: CreditRequest):
    payload = build_payload(data)

    return CreditPredictor.predict(payload)


@router.post("/explain")
def explain_credit_risk(data: CreditRequest):
    payload = build_payload(data)

    prediction = CreditPredictor.predict(payload)
    explanation = SHAPExplainer.explain(payload)

    return {
        "prediction": prediction,
        "explanation": explanation
    }