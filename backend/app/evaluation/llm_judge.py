import json
from backend.app.services.llm_service import client


class LLMJudge:

    @staticmethod
    def evaluate_answer(query: str, answer: str, sources: list[dict]):
        context = "\n\n".join(
            [f"[Source {s['source_id']}]\n{s['chunk']}" for s in sources]
        )

        prompt = f"""
You are an expert LLM evaluator.

Evaluate the answer using ONLY the retrieved context.

Return ONLY valid JSON.

Scoring:
- faithfulness: 0 to 1
- relevance: 0 to 1
- citation_quality: 0 to 1
- completeness: 0 to 1

Question:
{query}

Retrieved Context:
{context}

Answer:
{answer}

Return JSON:
{{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "citation_quality": 0.0,
  "completeness": 0.0,
  "overall_score": 0.0,
  "verdict": "PASS or FAIL",
  "reason": "short explanation"
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict LLM-as-Judge evaluator."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=500,
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except Exception:
            return {
                "error": "Invalid JSON returned by judge",
                "raw_output": content
            }