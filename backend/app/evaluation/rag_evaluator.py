import time
from backend.app.services.search_service import SearchService
from backend.app.services.llm_service import LLMService


class RAGEvaluator:

    @staticmethod
    def evaluate_query(query: str):
        start_time = time.time()

        sources = SearchService.search(query=query, top_k=5)

        answer = LLMService.generate_answer(
            query=query,
            sources=sources
        )

        latency = round(time.time() - start_time, 3)

        source_text = " ".join(
            [source["chunk"] for source in sources]
        ).lower()

        query_terms = [
            term for term in query.lower().split()
            if len(term) > 3
        ]

        matched_terms = [
            term for term in query_terms
            if term in source_text
        ]

        context_relevance = round(
            len(matched_terms) / max(len(query_terms), 1),
            3
        )

        citation_score = 1.0 if "[Source" in answer else 0.0

        answer_completeness = round(
            min(len(answer.split()) / 120, 1.0),
            3
        )

        overall_score = round(
            (context_relevance * 0.45)
            + (citation_score * 0.35)
            + (answer_completeness * 0.20),
            3
        )

        return {
            "query": query,
            "answer": answer,
            "metrics": {
                "context_relevance": context_relevance,
                "citation_score": citation_score,
                "answer_completeness": answer_completeness,
                "overall_rag_score": overall_score,
                "latency_seconds": latency
            },
            "sources_used": len(sources),
            "sources": sources
        }