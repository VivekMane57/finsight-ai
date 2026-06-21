import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


class LLMService:

    @staticmethod
    def generate_answer(query: str, sources: list[dict]) -> str:
        context = ""

        for source in sources:
            context += f"\n\n[Source {source['source_id']}]\n{source['chunk']}"

        prompt = f"""
You are FinSight AI, a senior financial analyst copilot.

Answer the user question using ONLY the provided financial document context.

Rules:
- Do not hallucinate.
- Use citations like [Source 1], [Source 2] after important claims.
- If the answer is not present, say: "I could not find this information in the uploaded document."
- Give a clear, structured financial answer.
- Use bullet points where useful.

User Question:
{query}

Retrieved Financial Context:
{context}

Final Answer:
"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "You are FinSight AI, a financial intelligence copilot for analysts, bankers, auditors, and credit risk teams."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700,
        )

        return response.choices[0].message.content