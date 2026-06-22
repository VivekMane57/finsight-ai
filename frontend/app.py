import streamlit as st
import requests

API_URL = "http://backend:8000"


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


st.set_page_config(
    page_title="FinSight AI",
    layout="wide"
)

st.title("FinSight AI")
st.caption("Financial Intelligence Copilot using Hybrid RAG + Azure OpenAI + LangGraph")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload Document",
    "Chat with Document",
    "Financial Summary",
    "KPI Dashboard",
    "Credit Risk",
    "Agentic Report",
    "Investment Analysis"
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
                col1.metric("Characters", data["characters"])
                col2.metric("Chunks", data["chunks"])
                col3.metric("Embedding Dim", data["embedding_dimension"])

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
            st.write(data["answer"])

            st.subheader("Retrieval Details")
            st.write("Method:", data["retrieval_method"])
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])
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
            st.write(data["summary"])

            st.subheader("Evidence")
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])
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
            kpis = data["kpis"]

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
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])

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
            st.write(data["credit_report"])

            st.subheader("Evidence")
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])
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
            st.write(" → ".join(data["agent_workflow"]))

            st.subheader("Final Multi-Agent Report")
            st.write(data["final_answer"])

            st.subheader("Evidence")
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])
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
            st.write(data["investment_report"])

            st.subheader("Evidence")
            st.write("Sources Used:", data["sources_used"])

            with st.expander("View Source Evidence"):
                for source in data["sources"]:
                    st.markdown(f"### Source {source['source_id']}")
                    st.write(source["chunk"])
        else:
            st.error(response.text)