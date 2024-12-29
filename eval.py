import streamlit as st
import pandas as pd
import time
from pandasai.smart_dataframe import SmartDataframe
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse

import pandasai.safe_libs.base_restricted_module as brm


def bypass_security():
    def wrapper(func):
        def wrapped_function(*args, **kwargs):
            # Bypass security checks
            return func(*args, **kwargs)

        return wrapped_function

    def wrap_function(func, *args, **kwargs):
        return wrapper(func)

    brm.BaseRestrictedModule._wrap_function = staticmethod(wrap_function)


# Initialize your LLM and SmartDataframe (adjust configuration as needed)
def setup_smart_dataframe():
    # Replace this with the path to your dataset
    data_path = "PURCHASE_ORDER_DATA_EXTRACT_2012-2015_0.csv"
    data = pd.read_csv(data_path)

    # Convert date fields to datetime
    date_columns = ["Creation Date", "Purchase Date"]
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )

    # Remove dollar signs and convert to numeric
    price_columns = ["Unit Price", "Total Price"]
    for col in price_columns:
        if col in data.columns:
            data[col] = data[col].replace(r"[\$,]", "", regex=True).astype(float)

    # Convert numeric fields
    numeric_columns = ["Quantity"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Convert categorical fields
    categorical_columns = [
        "Fiscal Year",
        "Acquisition Type",
        "Sub-Acquisition Type",
        "Acquisition Method",
        "Sub-Acquisition Method",
        "Department Name",
        "Supplier Qualifications",
        "Commodity Title",
        "Class Title",
        "Family Title",
        "Segment Title",
        "CalCard",
        "LPA Number",
        "Purchase Order Number",
        "Requisition Number",
        "Supplier Code",
        "Supplier Name",
        "Supplier Zip Code",
        "Item Name",
        "Item Description",
        "Classification Codes",
        "Normalized UNSPSC",
        "Class",
        "Family",
        "Segment",
    ]
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype("category")

    llm = ChatOpenAI(
        openai_api_key=st.secrets["openai_api_key"],  # Add your OpenAI API key here
        model="gpt-4o-2024-11-20",
        temperature=0,
    )

    metadata = """
    This dataset includes procurement records from California DGS spanning 2012-2015.
    Key attributes include:
    - Acquisition Methods (e.g., CMAS, Emergency Purchase)
    - Supplier Information
    - Price and Quantity Data
    """

    smart_df = SmartDataframe(
        data,
        name="Procurement Dataset",
        description=metadata,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "enable_cache": True,
            "verbose": True,
        },
    )

    return smart_df


# Predefined queries and answers
def get_test_queries():
    return [
        {
            "query": "What is the total spend in Fiscal Year 2013-2014?",
            "expected_answer": 42470675654.96,
        },
        {
            "query": "List the acquisition methods used in Fiscal Year 2013-2014.",
            "expected_answer": [
                "WSCA/Coop",
                "Informal Competitive",
                "Statewide Contract",
                "Services are specifically exempt by statute",
                "Fair and Reasonable",
                "State Programs",
                "SB/DVBE Option",
                "Formal Competitive",
                "CMAS",
                "NCB",
                "LCB",
                "Services are specifically exempt by policy",
                "Master Purchase/Price Agreement",
                "Master Service Agreement",
                "Emergency Purchase",
                "Software License Program",
                "Special Category Request (SCR)",
                "CRP",
                "Statement of Qualifications",
                "State Price Schedule",
            ],
        },
        {
            "query": "Who are the top 5 suppliers by total spend?",
            "expected_answer": {
                "Health Net Community Solutions, Inc.": 13587059000.06,
                "L.A. Care Health Plan": 11160129000.04,
                "Delta Dental of California": 8172038064.03,
                "Blue Cross of California Partnership Plan, Inc.": 7176560939.15,
                "Inland Empire Health Plan": 5438790004.0,
            },
        },
        {
            "query": "What is the average unit price for 'NON-IT Goods' in Fiscal Year 2013-2014?",
            "expected_answer": 14887.41,
        },
        {
            "query": "What is the total spend for suppliers with missing ZIP codes?",
            "expected_answer": 10537983325.66,
        },
        {
            "query": "What is the total spend for each fiscal year?",
            "expected_answer": {
                "2012-2013": 61912420939.42,
                "2013-2014": 42470675654.96,
                "2014-2015": 46860483903.82,
            },
        },
        {
            "query": "What are the top 5 departments by total spend?",
            "expected_answer": {
                "Health Care Services, Department of": 99759350736.42,
                "Public Health, Department of": 5621707893.98,
                "Social Services, Department of": 5565328198.27,
                "Corrections and Rehabilitation, Department of": 4711857451.29,
                "State Hospitals, Department of": 4545650046.42,
            },
        },
        {
            "query": "Which acquisition type accounts for the highest total spend?",
            "expected_answer": "NON-IT Services",
        },
        {
            "query": "What is the average quantity ordered for IT Goods?",
            "expected_answer": 91.36,
        },
        {
            "query": "Which supplier has the highest total spend?",
            "expected_answer": {
                "Supplier": "Health Net Community Solutions, Inc.",
                "Total Spend": 13587059000.06,
            },
        },
    ]


# Function to calculate evaluation metrics
def calculate_metrics(results):
    total_queries = len(results)
    correct_answers = sum(1 for r in results if r["Accuracy"] == "✅")
    error_count = sum(
        1 for r in results if r["Accuracy"] == "❌" and r["Response Time"] == "N/A"
    )
    response_times = [
        float(r["Response Time"].split()[0])
        for r in results
        if r["Response Time"] != "N/A"
    ]

    accuracy = (correct_answers / total_queries) * 100
    error_rate = (error_count / total_queries) * 100
    avg_response_time = (
        sum(response_times) / len(response_times) if response_times else 0
    )
    min_response_time = min(response_times, default=0)
    max_response_time = max(response_times, default=0)

    return {
        "Accuracy (%)": accuracy,
        "Error Rate (%)": error_rate,
        "Avg Response Time (s)": avg_response_time,
        "Min Response Time (s)": min_response_time,
        "Max Response Time (s)": max_response_time,
    }


# Function to evaluate model performance
def evaluate_queries(smart_df, queries):
    results = []

    def is_close(a, b, tol=0.01):
        """Helper function to allow tolerance in numeric comparisons."""
        try:
            return abs(float(a) - float(b)) <= tol
        except (ValueError, TypeError):
            return False

    for test in queries:
        query = test["query"]
        expected = test["expected_answer"]

        start_time = time.time()
        try:
            answer = smart_df.chat(query)
            response_time = time.time() - start_time

            # Check accuracy with tolerance for numeric comparisons
            if isinstance(expected, (list, dict)):
                accuracy = answer == expected
            elif isinstance(expected, (int, float)):
                accuracy = is_close(answer, expected)
            elif "chart" in str(expected).lower():
                accuracy = isinstance(answer, str) and answer.startswith(
                    "data:image/png;base64,"
                )
            else:
                accuracy = str(answer).strip() == str(expected).strip()

            results.append(
                {
                    "Query": query,
                    "Expected": expected,
                    "Answer": answer,
                    "Accuracy": "✅" if accuracy else "❌",
                    "Response Time": f"{response_time:.2f} seconds",
                }
            )

        except Exception as e:
            results.append(
                {
                    "Query": query,
                    "Expected": expected,
                    "Answer": str(e),
                    "Accuracy": "❌",
                    "Response Time": "N/A",
                }
            )

    return results


# Main app logic
def main():
    st.title("Q/A System Evaluation Tool")

    smart_df = setup_smart_dataframe()
    queries = get_test_queries()

    if st.button("Run Evaluation"):
        st.markdown("## Evaluation Results")

        results = evaluate_queries(smart_df, queries)
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Calculate metrics
        metrics = calculate_metrics(results)
        st.markdown("### Metrics Summary")
        st.write(metrics)

        # Optionally save results to a file
        results_df.to_csv("evaluation_results.csv", index=False)
        st.success(
            "Evaluation completed and results saved to 'evaluation_results.csv'."
        )


if __name__ == "__main__":
    main()
