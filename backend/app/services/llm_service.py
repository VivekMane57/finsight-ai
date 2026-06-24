import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langsmith import traceable

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)


class LLMService:

    @staticmethod
    def _build_context(sources: list[dict]) -> str:
        context = ""

        for source in sources:
            context += f"\n\n[Source {source['source_id']}]\n{source['chunk']}"

        return context

    @staticmethod
    @traceable(name="FinSight Generate Answer")
    def generate_answer(query: str, sources: list[dict]) -> str:
        context = LLMService._build_context(sources)

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

    @staticmethod
    @traceable(name="FinSight Financial Summary")
    def generate_financial_summary(sources: list[dict]) -> str:
        context = LLMService._build_context(sources)

        prompt = f"""
You are FinSight AI, a senior equity research analyst.

Create a professional financial analyst report using ONLY the provided context.

Important:
- Do not say information is unavailable unless the context truly has no related evidence.
- Use citations like [Source 1], [Source 2].
- If exact numbers are not available, summarize trends and evidence qualitatively.
- Do not hallucinate numbers.

Report Format:

1. Executive Summary
2. Revenue Overview
3. Profitability Analysis
4. Balance Sheet Highlights
5. Cash Flow Analysis
6. Key Business Risks
7. Investment / Credit Takeaways

Retrieved Financial Context:
{context}

Final Financial Analyst Report:
"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior financial analyst generating source-grounded financial reports."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content

    @staticmethod
    @traceable(name="FinSight Credit Risk Analysis")
    def generate_credit_risk_analysis(sources: list[dict]) -> str:
        context = LLMService._build_context(sources)

        prompt = f"""
You are a Senior Credit Risk Analyst working at a global bank.

Analyze the company's credit risk using ONLY the provided financial document context.

Return a structured credit risk report.

Include:

1. Credit Score out of 100
2. Risk Level: Low, Moderate, or High
3. Revenue Assessment
4. Profitability Assessment
5. Liquidity Assessment
6. Cash Flow Assessment
7. Key Strengths
8. Key Weaknesses
9. Credit Recommendation: APPROVE, APPROVE WITH CONDITIONS, or REJECT

Rules:
- Do not hallucinate.
- Use citations like [Source 1], [Source 2].
- If exact data is unavailable, give qualitative assessment from available evidence.
- Be professional and bank/lending focused.

Financial Context:
{context}

Final Credit Risk Report:
"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior credit risk analyst for banks and NBFC lending teams."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content

    @staticmethod
    @traceable(name="FinSight Custom Analysis")
    def generate_custom_analysis(prompt: str, sources: list[dict]) -> str:
        context = LLMService._build_context(sources)

        final_prompt = f"""
{prompt}

Rules:
- Use ONLY the provided financial context.
- Do not hallucinate.
- Use citations like [Source 1], [Source 2].
- If exact valuation data is unavailable, say that valuation is qualitative.
- Do not provide personal financial advice.

Financial Context:
{context}

Final Report:
"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior equity research analyst and investment analyst."
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content

    @staticmethod
    @traceable(name="FinSight KPI Extraction")
    def extract_kpis(sources: list[dict]) -> str:
        context = LLMService._build_context(sources)

        prompt = f"""
You are a financial KPI extraction engine.

Extract financial KPIs from the provided annual report context.

Return ONLY valid JSON. Do not include markdown. Do not include explanations.

JSON fields:
{{
  "company": null,
  "fiscal_year": null,
  "currency": null,
  "revenue": null,
  "revenue_growth": null,
  "gross_profit": null,
  "operating_profit": null,
  "net_profit": null,
  "operating_margin": null,
  "net_margin": null,
  "eps_basic": null,
  "operating_cash_flow": null,
  "free_cash_flow": null,
  "total_assets": null,
  "total_liabilities": null
}}

Rules:
- Use null if a value is not available.
- Do not hallucinate.
- Numbers must be numeric, not strings.
- Currency should be like "USD Millions".
- Percentages should be numeric only, e.g. 21.1.

Financial Context:
{context}

Valid JSON:
"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "You extract financial KPIs as strict valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=900,
        )

        return response.choices[0].message.content