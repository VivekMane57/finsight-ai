from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService


class InvestmentAgent:

    @staticmethod
    def analyze():

        sources = SearchService.search(
            """
            revenue growth profitability operating margin
            net margin cash flow risks market position
            investment outlook
            """,
            top_k=10
        )

        prompt = """
Act as a Senior Equity Research Analyst.

Generate:

1. Investment Score (0-100)
2. Bull Case
3. Bear Case
4. Growth Drivers
5. Risks
6. Valuation Perspective
7. Final Recommendation

Recommendation must be one of:

BUY
HOLD
SELL

Use only provided evidence.
"""

        report = LLMService.generate_custom_analysis(
            prompt,
            sources
        )

        return {
            "investment_report": report,
            "sources_used": len(sources),
            "sources": sources
        }