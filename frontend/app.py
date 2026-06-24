import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://backend:8000")


def fmt_money(value, currency="USD Millions"):
    if value is None:
        return "N/A"
    return f"${value}M" if "USD" in str(currency) else str(value)


def fmt_percent(value):
    if value is None:
        return "N/A"
    return f"{value}%"


def fmt_value(value):
    if value is None:
        return "N/A"
    return str(value)


def show_sources(sources):
    with st.expander("View Source Evidence"):
        for source in sources:
            st.markdown(f"### Source {source.get('source_id')}")
            if "rerank_score" in source:
                st.caption(f"CrossEncoder Rerank Score: {source.get('rerank_score')}")
            st.write(source.get("chunk"))


st.set_page_config(
    page_title="FinSight AI",
    layout="wide"
)

st.title("FinSight AI")
st.caption(
    "Financial Intelligence Copilot using Hybrid RAG + CrossEncoder + Azure OpenAI + LangGraph + LangSmith + XGBoost + SHAP"
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Upload Document",
    "Chat with Document",
    "Financial Summary",
    "KPI Dashboard",
    "Credit Risk",
    "Agentic Report",
    "Investment Analysis",
    "LLM Evaluation",
    "ML Credit Risk"
])

with tab1:
    st.header("Upload Financial Document")

    uploaded_file = st.file_uploader(
        "Upload annual report PDF",
        type=["pdf"]
    )

    if st.button("Process Document"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }

            with st.spinner("Processing document..."):
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files=files
                )

            if response.status_code == 200:
                data = response.json()
                st.success("Document indexed successfully.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Characters", data.get("characters"))
                col2.metric("Chunks", data.get("chunks"))
                col3.metric("Embedding Dim", data.get("embedding_dimension"))

                st.json(data)
            else:
                st.error(response.text)

with tab2:
    st.header("Ask Financial Questions")

    query = st.text_area(
        "Ask a question",
        value="What are the major business risks?"
    )

    if st.button("Ask FinSight AI"):
        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{API_URL}/chat/query",
                json={"query": query}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data.get("answer"))

            st.subheader("Retrieval Details")
            st.write("Method:", data.get("retrieval_method"))
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))
        else:
            st.error(response.text)

with tab3:
    st.header("AI Financial Summary")

    st.write(
        "Generate an analyst-style financial summary from the uploaded annual report."
    )

    if st.button("Generate Financial Summary"):
        with st.spinner("Analyzing financial document..."):
            response = requests.post(
                f"{API_URL}/analysis/financial-summary"
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Financial Analyst Summary")
            st.write(data.get("summary"))

            st.subheader("Evidence")
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))
        else:
            st.error(response.text)

with tab4:
    st.header("Auto Financial KPI Dashboard")

    st.write(
        "Automatically extract financial KPIs from the uploaded annual report using RAG + Azure OpenAI."
    )

    if st.button("Extract KPIs"):
        with st.spinner("Extracting KPIs from document..."):
            response = requests.get(
                f"{API_URL}/kpi/financial-kpis"
            )

        if response.status_code == 200:
            data = response.json()
            kpis = data.get("kpis", {})

            currency = kpis.get("currency") or "USD Millions"

            st.subheader(
                f"{kpis.get('company') or 'Company'} - FY {kpis.get('fiscal_year') or 'N/A'}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Revenue",
                fmt_money(kpis.get("revenue"), currency),
                fmt_percent(kpis.get("revenue_growth"))
            )

            col2.metric(
                "Net Profit",
                fmt_money(kpis.get("net_profit"), currency)
            )

            col3.metric(
                "Operating Margin",
                fmt_percent(kpis.get("operating_margin"))
            )

            col4.metric(
                "Net Margin",
                fmt_percent(kpis.get("net_margin"))
            )

            col5, col6, col7, col8 = st.columns(4)

            col5.metric(
                "Operating Profit",
                fmt_money(kpis.get("operating_profit"), currency)
            )

            col6.metric(
                "Gross Profit",
                fmt_money(kpis.get("gross_profit"), currency)
            )

            col7.metric(
                "Basic EPS",
                fmt_value(kpis.get("eps_basic"))
            )

            col8.metric(
                "Operating Cash Flow",
                fmt_money(kpis.get("operating_cash_flow"), currency)
            )

            st.subheader("Revenue and Profit Overview")

            chart_data = {
                "Metric": [],
                "Value": []
            }

            for metric_key, metric_name in [
                ("revenue", "Revenue"),
                ("gross_profit", "Gross Profit"),
                ("operating_profit", "Operating Profit"),
                ("net_profit", "Net Profit")
            ]:
                value = kpis.get(metric_key)
                if value is not None:
                    chart_data["Metric"].append(metric_name)
                    chart_data["Value"].append(value)

            if chart_data["Value"]:
                st.bar_chart(
                    data=chart_data,
                    x="Metric",
                    y="Value"
                )
            else:
                st.warning("No chartable profit/revenue KPI values found.")

            st.subheader("Margin Overview")

            margin_data = {
                "Metric": [],
                "Value": []
            }

            for metric_key, metric_name in [
                ("operating_margin", "Operating Margin"),
                ("net_margin", "Net Margin")
            ]:
                value = kpis.get(metric_key)
                if value is not None:
                    margin_data["Metric"].append(metric_name)
                    margin_data["Value"].append(value)

            if margin_data["Value"]:
                st.bar_chart(
                    data=margin_data,
                    x="Metric",
                    y="Value"
                )
            else:
                st.warning("No margin KPI values found.")

            st.subheader("Balance Sheet / Cash Flow KPIs")

            col9, col10, col11 = st.columns(3)

            col9.metric(
                "Total Assets",
                fmt_money(kpis.get("total_assets"), currency)
            )

            col10.metric(
                "Total Liabilities",
                fmt_money(kpis.get("total_liabilities"), currency)
            )

            col11.metric(
                "Free Cash Flow",
                fmt_money(kpis.get("free_cash_flow"), currency)
            )

            st.subheader("Raw Extracted KPI JSON")
            st.json(kpis)

            st.subheader("Evidence")
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))

        else:
            st.error(response.text)

with tab5:
    st.header("AI Credit Risk Analysis")

    st.write(
        "Generate a lender-style credit risk assessment from the uploaded annual report."
    )

    if st.button("Analyze Credit Risk"):
        with st.spinner("Analyzing credit risk..."):
            response = requests.post(
                f"{API_URL}/analysis/credit-risk"
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Credit Risk Report")
            st.write(data.get("credit_report"))

            st.subheader("Evidence")
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))
        else:
            st.error(response.text)

with tab6:
    st.header("LangGraph Agentic Financial Report")

    st.write(
        "Run a multi-agent financial intelligence workflow using Retrieval, Summary, Credit Risk, and Final Analyst agents."
    )

    agent_query = st.text_area(
        "Agent Query",
        value="Analyze the financial health and credit risk of the company."
    )

    if st.button("Run Multi-Agent Analysis"):
        with st.spinner("Running LangGraph agents..."):
            response = requests.post(
                f"{API_URL}/agents/financial-intelligence",
                json={"query": agent_query}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Agent Workflow")
            st.write(" → ".join(data.get("agent_workflow", [])))

            st.subheader("Final Multi-Agent Report")
            st.write(data.get("final_answer"))

            st.subheader("Evidence")
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))
        else:
            st.error(response.text)

with tab7:
    st.header("AI Investment Analysis")

    st.write(
        "Generate an analyst-style investment recommendation from the uploaded annual report."
    )

    if st.button("Generate Investment Report"):
        with st.spinner("Generating investment analysis..."):
            response = requests.post(
                f"{API_URL}/analysis/investment-analysis"
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Investment Recommendation Report")
            st.write(data.get("investment_report"))

            st.subheader("Evidence")
            st.write("Sources Used:", data.get("sources_used"))

            show_sources(data.get("sources", []))
        else:
            st.error(response.text)

with tab8:
    st.header("LLM Evaluation")

    st.write(
        "Evaluate RAG response quality using Custom RAG Evaluation, RAGAS, and LLM-as-Judge."
    )

    eval_query = st.text_area(
        "Evaluation Query",
        value="What are the major business risks?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Custom RAG Evaluation"):
            with st.spinner("Running custom RAG evaluation..."):
                response = requests.post(
                    f"{API_URL}/evaluation/rag",
                    json={"query": eval_query}
                )

            if response.status_code == 200:
                data = response.json()

                st.subheader("Custom Evaluation Metrics")
                metrics = data.get("metrics", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Context Relevance", metrics.get("context_relevance"))
                c2.metric("Citation Score", metrics.get("citation_score"))
                c3.metric("Completeness", metrics.get("answer_completeness"))
                c4.metric("Overall RAG Score", metrics.get("overall_rag_score"))

                st.metric("Latency Seconds", metrics.get("latency_seconds"))

                st.subheader("Answer")
                st.write(data.get("answer"))

                show_sources(data.get("sources", []))
            else:
                st.error(response.text)

    with col2:
        if st.button("Run LLM-as-Judge"):
            with st.spinner("Running LLM-as-Judge evaluation..."):
                response = requests.post(
                    f"{API_URL}/evaluation/judge",
                    json={"query": eval_query}
                )

            if response.status_code == 200:
                data = response.json()
                judge = data.get("judge_result", {})

                st.subheader("LLM-as-Judge Result")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness", judge.get("faithfulness"))
                c2.metric("Relevance", judge.get("relevance"))
                c3.metric("Citation Quality", judge.get("citation_quality"))
                c4.metric("Completeness", judge.get("completeness"))

                st.metric("Overall Score", judge.get("overall_score"))
                st.success(f"Verdict: {judge.get('verdict')}")
                st.write(judge.get("reason"))

                st.subheader("Answer")
                st.write(data.get("answer"))

                show_sources(data.get("sources", []))
            else:
                st.error(response.text)

    with col3:
        if st.button("Run RAGAS Evaluation"):
            with st.spinner("Running RAGAS evaluation..."):
                response = requests.post(
                    f"{API_URL}/ragas/evaluate",
                    json={"query": eval_query}
                )

            if response.status_code == 200:
                data = response.json()

                st.subheader("RAGAS Scores")
                scores = data.get("scores", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Faithfulness", scores.get("faithfulness"))
                c2.metric("Answer Relevancy", scores.get("answer_relevancy"))
                c3.metric("Context Precision", scores.get("context_precision"))
                c4.metric("Context Recall", scores.get("context_recall"))

                st.subheader("Answer")
                st.write(data.get("answer"))

                with st.expander("Raw RAGAS Output"):
                    st.json(scores)
            else:
                st.error(response.text)

with tab9:
    st.header("ML Credit Risk Prediction")

    st.write(
        "Predict default risk using an XGBoost credit risk model and explain predictions using SHAP."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.number_input("Job Level", min_value=0, max_value=3, value=2)

    with col2:
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox(
            "Saving Accounts",
            ["little", "moderate", "quite rich", "rich"]
        )
        checking_account = st.selectbox(
            "Checking Account",
            ["little", "moderate", "rich"]
        )

    with col3:
        credit_amount = st.number_input(
            "Credit Amount",
            min_value=100,
            max_value=50000,
            value=5000
        )
        duration = st.number_input(
            "Duration Months",
            min_value=1,
            max_value=120,
            value=24
        )
        purpose = st.selectbox(
            "Purpose",
            [
                "car",
                "furniture/equipment",
                "radio/TV",
                "domestic appliances",
                "repairs",
                "education",
                "business",
                "vacation/others"
            ]
        )

    payload = {
        "Age": int(age),
        "Sex": sex,
        "Job": int(job),
        "Housing": housing,
        "Saving_accounts": saving_accounts,
        "Checking_account": checking_account,
        "Credit_amount": int(credit_amount),
        "Duration": int(duration),
        "Purpose": purpose
    }

    col_predict, col_explain = st.columns(2)

    with col_predict:
        if st.button("Predict Credit Risk"):
            with st.spinner("Running XGBoost prediction..."):
                response = requests.post(
                    f"{API_URL}/credit/predict",
                    json=payload
                )

            if response.status_code == 200:
                data = response.json()

                st.subheader("Prediction Result")

                c1, c2, c3 = st.columns(3)
                c1.metric("Credit Score", data.get("credit_score"))
                c2.metric("Default Probability", data.get("default_probability"))
                c3.metric("Risk Level", data.get("risk_level"))

                st.json(data)
            else:
                st.error(response.text)

    with col_explain:
        if st.button("Explain with SHAP"):
            with st.spinner("Generating SHAP explanation..."):
                response = requests.post(
                    f"{API_URL}/credit/explain",
                    json=payload
                )

            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", {})
                explanation = data.get("explanation", {})
                factors = explanation.get("top_risk_factors", [])

                st.subheader("Prediction")
                c1, c2, c3 = st.columns(3)
                c1.metric("Credit Score", prediction.get("credit_score"))
                c2.metric("Default Probability", prediction.get("default_probability"))
                c3.metric("Risk Level", prediction.get("risk_level"))

                st.subheader("Top SHAP Risk Factors")

                if factors:
                    chart_data = {
                        "Feature": [item["feature"] for item in factors],
                        "Impact": [item["impact"] for item in factors]
                    }

                    st.bar_chart(
                        data=chart_data,
                        x="Feature",
                        y="Impact"
                    )

                    st.json(factors)
                else:
                    st.warning("No SHAP factors returned.")

                st.caption(explanation.get("model"))
            else:
                st.error(response.text)