import os

import matplotlib.pyplot as plt
import pandas as pd
import pandasai.safe_libs.base_restricted_module as brm
import streamlit as st
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.smart_dataframe import SmartDataframe
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def bypass_security():
    def wrapper(func):
        def wrapped_function(*args, **kwargs):
            # Bypass security checks
            return func(*args, **kwargs)

        return wrapped_function

    def wrap_function(func, *args, **kwargs):
        return wrapper(func)

    brm.BaseRestrictedModule._wrap_function = staticmethod(wrap_function)


bypass_security()


# Load and clean the procurement dataset
@st.cache_data
def load_and_clean_data():
    data_path = "PURCHASE_ORDER_DATA_EXTRACT_2012-2015_0.csv"

    try:
        # Check if CSV file exists
        if os.path.exists(data_path):
            print("Loading data from CSV file...")
            data = pd.read_csv(data_path)
        else:
            print("CSV file not found. Loading data from MongoDB...")
            # MongoDB connection URI
            uri = f"mongodb+srv://{st.secrets['db_username']}:{st.secrets['db_password']}@cluster0.nzc35.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            # Connect to MongoDB
            client = MongoClient(uri, server_api=ServerApi("1"))

            # Fetch data from MongoDB
            db = client["purchase_orders"]
            collection = db["order_data"]
            data = pd.DataFrame(list(collection.find()))

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

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

    return data


# Cache the LangChain LLM setup
@st.cache_resource
def setup_pandasai():
    openai_api_key = st.secrets[
        "openai_api_key"
    ]  # Store your API key in Streamlit secrets
    langchain_llm = ChatOpenAI(
        openai_api_key=openai_api_key, model="gpt-4o-2024-11-20", temperature=0
    )
    return langchain_llm


# Define dataset description and column mapping
dataset_metadata = """
This dataset includes detailed procurement records from the California Department of General Services (DGS) spanning fiscal years 2012 to 2015. It provides insights into supplier information, item classification, and acquisition methods.

Metadata Overview:
- Temporal Coverage: Fiscal Years 2012–2015.
- Primary Fields:
  - Supplier information.
  - UNSPSC item classifications.
  - Acquisition methods including competitive bids, emergency purchases, policy exemptions, etc.
- Supplier Qualifications: Includes certifications like Small Business (SB), Disabled Veteran Business Enterprise (DVBE), Non-Profit (NP), and Micro-Business (MB).

Data Types:
1. Date Fields:
   - 'Creation Date': System-generated date.
   - 'Purchase Date': User-entered date; may be backdated.
2. Numeric Fields:
   - 'Unit Price', 'Total Price', 'Quantity' (floats).
3. Categorical Fields:
   - 'Fiscal Year', 'Acquisition Type', 'Acquisition Method', 'Department Name'.
   - UNSPSC classification hierarchy: Segment, Family, Class, Commodity Titles.
4. Boolean:
   - 'CalCard' (Yes/No).

For further clarification on acquisition methods (e.g., CMAS, CRP, Fair and Reasonable), refer to supplemental acquisition method documentation.
Core Fields:
- Creation Date: System-generated date.
- Purchase Date: Entered by users, potentially backdated; 'Creation Date' is the primary reference.
- Fiscal Year: Derived from 'Creation Date.' California's fiscal year starts on July 1 and ends on June 30.
- Acquisition Type: Categorized into Non-IT Goods, Non-IT Services, IT Goods, IT Services.
- Acquisition Method: Type of acquisition used. Detailed methods include CMAS, Statewide Contracts, etc. Refer to the data dictionary for specifics.
- Department Name: Purchasing department's name.

Supplier Information:
- Supplier Code: Unique identifier for suppliers.
- Supplier Name: As registered with the state.
- Supplier Qualifications: Indicates SB, SBE, DVBE, NP, MB certifications.

Item Details:
- Item Name/Description: Details about purchased items.
- Quantity: Number of items.
- Unit Price: Price per unit.
- Total Price: Excludes taxes or shipping costs.

UNSPSC Classifications:
- Segment, Family, Class, Commodity Title: Categorized according to UNSPSC v14.

CalCard:
- Indicates whether the state credit card was used (Yes/No).

Notes:
- Field Normalization: Fields like 'Department Name' and 'Supplier Code' are normalized.
- Classification Codes: Correlates to UNSPSC v14, ensuring accurate item categorization.
"""


def display_example_questions():
    st.markdown(
        """### Example Questions to Try:

1. What is the total spend in Fiscal Year 2013-2014?
2. List the acquisition methods used in Fiscal Year 2013-2014.
3. Who are the top 5 suppliers by total spend?
4. What is the average unit price for 'NON-IT Goods' in Fiscal Year 2013-2014?
5. Which supplier has the highest number of purchase orders?
6. What is the average spend per supplier in Fiscal Year 2012-2013?
7. Which department made the single highest-value purchase, and what was the amount?
8. Generate a bar chart showing total spend by acquisition type for Fiscal Year 2013-2014.
9. Generate a pie chart of total spend by department in Fiscal Year 2014-2015.
10. How does the total spend vary across acquisition types in Fiscal Year 2014-2015?

Feel free to type these questions or explore your own!"""
    )


# Main application logic
def main():
    st.title("Procurement Data Query Assistant")

    st.markdown(
        """
        This application allows you to interact with procurement data from 2012-2015.
        Type your queries in natural language or request visualizations.
    """
    )

    data = load_and_clean_data()
    llm = setup_pandasai()

    smart_df = SmartDataframe(
        data,
        name="Procurement Dataset",
        description=dataset_metadata,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "enable_cache": True,
            "verbose": True,
            "save_charts": False,
            "open_charts": False,
            "custom_whitelisted_dependencies": ["scikit-learn", "plotly"],
        },
    )

    with st.expander("View Sample Data (10 rows)"):
        st.write(data.head(10))
    # Display example questions
    with st.expander("Example Questions to Inspire You"):
        display_example_questions()

    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Processing your query..."):
            try:
                # Get both the result and the generated code
                result = smart_df.chat(query)
                generated_code = smart_df.last_code_generated

                # Display the result
                st.success("Query Processed!")
                st.write("### Result:")

                # Handle different result types
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, str):
                    if result.startswith("data:image/png;base64,"):
                        st.image(result.split(",")[1])
                    elif result.endswith(".png"):
                        st.image(
                            result,
                        )
                    else:
                        st.write(result)
                elif isinstance(result, dict) and "type" in result:
                    # Handle PandasAI response dictionary
                    if result["type"] == "plot":
                        if isinstance(result["value"], str):
                            if result["value"].startswith("data:image/png;base64,"):
                                st.image(
                                    result["value"].split(",")[1],
                                )
                            else:
                                st.image(
                                    result["value"],
                                )
                    elif result["type"] == "dataframe":
                        st.dataframe(result["value"])
                    else:
                        st.write(result["value"])
                else:
                    st.write(result)

                # Display the generated code in an expander
                with st.expander("View Generated Code"):
                    st.code(generated_code, language="python")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
