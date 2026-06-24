import os
import math
from datasets import Dataset
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

load_dotenv()


class RagasEvaluator:

    @staticmethod
    def _azure_llm():
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=0
        )

    @staticmethod
    def _azure_embeddings():
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )

    @staticmethod
    def _clean_value(value):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    @staticmethod
    def _clean_dict(data: dict):
        return {
            key: RagasEvaluator._clean_value(value)
            for key, value in data.items()
        }

    @staticmethod
    def evaluate_response(
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = ""
    ):
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth]
            }
        )

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=RagasEvaluator._azure_llm(),
            embeddings=RagasEvaluator._azure_embeddings(),
            raise_exceptions=False
        )

        scores = result.to_pandas().to_dict(orient="records")[0]

        return RagasEvaluator._clean_dict(scores)