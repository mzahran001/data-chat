import json
import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import pandasai.safe_libs.base_restricted_module as brm
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.smart_dataframe import SmartDataframe
from pydantic import BaseModel, Field

from mongodb import MongoDBHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture essential log messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs will be output to the console
    ],
)

logger = logging.getLogger(__name__)


class MongoDBQueryFilter(BaseModel):
    """Schema for MongoDB query filters."""

    filter: Dict[str, Any] = Field(
        ..., description="The MongoDB query filter in dictionary format."
    )

    class Config:
        title = "MongoDBQueryFilter"
        description = "A filter schema for querying MongoDB collections."


class LangChainQueryTranslator:
    """Translates natural language queries into MongoDB queries using LangChain with structured output."""

    def __init__(self, openai_api_key, dataset_metadata):
        """Initialize the LangChain LLM with structured output."""
        logger.info("Initializing LangChainQueryTranslator with structured output.")
        try:
            # Initialize the ChatOpenAI LLM
            self.llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model="gpt-4o",  # Keeping the original model name
                temperature=0,
            )
            logger.info("ChatOpenAI initialized successfully.")

            # Define the Pydantic output parser
            self.output_parser = PydanticOutputParser(
                pydantic_object=MongoDBQueryFilter
            )
            logger.info("PydanticOutputParser initialized successfully.")

            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an intelligent assistant designed to translate natural language queries into MongoDB query filters in JSON format.\n\n"
                        "Here is the detailed schema of the dataset you will be querying:\n{dataset_metadata}\n\n"
                        "Translate the following user query into a MongoDB query filter that accurately retrieves the desired data. "
                        "**If the query does not specify any filterable condition (like Fiscal Year, Department Name, Supplier Name, etc.), return an empty filter (`{{}}`).** "
                        "The output must be a JSON object with a top-level 'filter' key. **Do not include any markdown formatting or code blocks.**\n\n"
                        "{format_instructions}",
                    ),
                    ("human", "{user_query}"),
                ]
            ).partial(format_instructions=self.output_parser.get_format_instructions())

            logger.info("PromptTemplate created successfully.")

            # Create the chain: prompt -> LLM -> parser
            self.chain = self.prompt | self.llm | self.output_parser
            logger.info(
                "LangChain chain with structured output constructed successfully."
            )
        except Exception as e:
            logger.error(f"Error initializing LangChainQueryTranslator: {e}")
            raise

    def translate(self, user_query, dataset_metadata):
        """Translate a natural language query to a MongoDB query filter.

        Args:
            user_query (str): The user's natural language query.
            dataset_metadata (str): Detailed schema information.

        Returns:
            dict: MongoDB query filter.
        """
        logger.info("Translating user query into MongoDB filter.")
        try:
            response = self.chain.invoke(
                {"user_query": user_query, "dataset_metadata": dataset_metadata}
            )
            logger.info("Structured response received from LangChain.")

            # Access the parsed output
            mongo_query = response.filter
            logger.info(f"Generated MongoDB query: {mongo_query}")

            if not isinstance(mongo_query, dict):
                raise ValueError("Translated query is not a valid dictionary.")
            logger.info("Query translated successfully.")
            return mongo_query

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to empty filter if something invalid is returned
            logger.error(
                f"Translator returned invalid JSON, defaulting to empty filter: {e}"
            )
            return {}
        except Exception as e:
            logger.error(f"Error translating query: {e}")
            st.error(f"Error translating query: {e}")
            return {}


class SecurityBypass:
    """Utility class to bypass PandasAI security restrictions."""

    @staticmethod
    def bypass_security():
        """Bypass PandasAI security checks for custom visualizations."""
        logger.info("Applying PandasAI security bypass.")

        def wrapper(func):
            def wrapped_function(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_function

        def wrap_function(func, *args, **kwargs):
            return wrapper(func)

        brm.BaseRestrictedModule._wrap_function = staticmethod(wrap_function)
        logger.info("PandasAI security bypass applied successfully.")


class LangChainSetup:
    """Handles LangChain setup and configuration."""

    @staticmethod
    @st.cache_resource
    def setup_pandasai(openai_api_key):
        """Set up LangChain with PandasAI integration."""
        logger.info("Setting up LangChain with PandasAI integration.")
        try:
            langchain_llm = ChatOpenAI(
                api_key=openai_api_key, model="gpt-4o", temperature=0
            )
            logger.info("LangChain LLM setup successfully.")
            return langchain_llm
        except Exception as e:
            logger.error(f"Error setting up LangChain: {e}")
            raise


class DataLoader:
    """Handles data loading and cleaning operations."""

    @staticmethod
    def clean_data(data):
        """Clean and process the data."""
        logger.info("Cleaning data.")
        try:
            # Convert date fields to datetime
            date_columns = ["Creation Date", "Purchase Date"]
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col], errors="coerce").dt.strftime(
                        "%Y-%m-%d"
                    )
                    logger.info(f"Converted column '{col}' to datetime.")

            # Remove dollar signs and convert to numeric
            price_columns = ["Unit Price", "Total Price"]
            for col in price_columns:
                if col in data.columns:
                    data[col] = (
                        data[col].replace(r"[\$,]", "", regex=True).astype(float)
                    )
                    logger.info(f"Converted column '{col}' to float.")

            # Convert numeric fields
            numeric_columns = ["Quantity"]
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
                    logger.info(f"Converted column '{col}' to numeric.")

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
                    logger.info(f"Converted column '{col}' to category.")

            logger.info("Data cleaning completed successfully.")
            return data
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_data(_mongo_handler, mongo_query):
        """Load data from MongoDB based on the provided query and perform cleaning operations.

        Args:
            _mongo_handler (MongoDBHandler): Instance of MongoDBHandler. _ prefix to prevent caching collision.
            mongo_query (dict): MongoDB query filter.

        Returns:
            pd.DataFrame: Cleaned data.
        """
        logger.info("Loading data from MongoDB.")
        try:
            st.info("Executing MongoDB query...")
            data = pd.DataFrame(list(_mongo_handler.collection.find(mongo_query)))
            logger.info(f"Data fetched from MongoDB: {data.shape[0]} records.")

            if "_id" in data.columns:
                data = data.drop(columns=["_id"])
                logger.info("Dropped '_id' column from data.")

            if data.empty:
                st.warning("No data found for the given query.")
                logger.warning("No data found for the provided MongoDB query.")
                return pd.DataFrame()

            cleaned_data = DataLoader.clean_data(data)
            logger.info("Data loaded and cleaned successfully.")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            raise


class ProcurementApp:
    """Streamlit-based procurement data query assistant."""

    def __init__(self):
        logger.info("Initializing ProcurementApp.")
        self.dataset_metadata = """
This dataset includes detailed procurement records from the California Department of General Services (DGS) spanning fiscal years 2012 to 2015. It provides insights into supplier information, item classification, and acquisition methods.

**Metadata Overview:**
- **Temporal Coverage**: Fiscal Years 2012–2015.
- **Primary Fields**:
  - **Supplier Information**:
    - `Supplier Code` (String): Unique identifier for suppliers.
    - `Supplier Name` (String): Registered name of the supplier.
    - `Supplier Qualifications` (Categorical): Certifications like SB, DVBE, NP, MB.
    - `Supplier Zip Code` (String): ZIP code of the supplier's address.
  - **Acquisition Details**:
    - `Fiscal Year` (String): Derived from 'Creation Date.' Fiscal Year starts on July 1 and ends on June 30.
    - `Acquisition Type` (Categorical): Non-IT Goods, Non-IT Services, IT Goods, IT Services.
    - `Acquisition Method` (Categorical): Methods like CMAS, Statewide Contracts, Emergency Purchases, etc.
  - **Item Details**:
    - `Item Name` (String): Name of the purchased item.
    - `Item Description` (String): Description of the purchased item.
    - `Quantity` (Integer): Number of items purchased.
    - `Unit Price` (Float): Price per unit.
    - `Total Price` (Float): Total price excluding taxes and shipping.
    - `Normalized UNSPSC` (String): UNSPSC classification codes.
  - **Date Fields**:
    - `Creation Date` (Date): System-generated date of record creation.
    - `Purchase Date` (Date): User-entered date of purchase; may be backdated.
  - **Department Details**:
    - `Department Name` (String): Name of the purchasing department.
  - **Miscellaneous**:
    - `CalCard` (Boolean): Indicates if a state credit card was used.
    - `LPA Number`, `Purchase Order Number`, `Requisition Number` (String): Various identifiers.

**Data Types:**
1. **Date Fields**:
   - `Creation Date`, `Purchase Date`
2. **Numeric Fields**:
   - `Unit Price`, `Total Price` (Float)
   - `Quantity` (Integer)
3. **Categorical Fields**:
   - `Fiscal Year`, `Acquisition Type`, `Acquisition Method`, `Department Name`
   - `Supplier Qualifications`, `Normalized UNSPSC`
4. **Boolean Fields**:
   - `CalCard` (Yes/No)

**Relationships and Constraints:**
- **Supplier Relationships**: Each `Purchase Order` is linked to a `Supplier` via `Supplier Code`.
- **Fiscal Year Calculation**: Derived from `Creation Date`.
- **UNSPSC Classification**: Hierarchical structure comprising Segment, Family, Class, and Commodity Titles.
"""

    def display_example_questions(self):
        """Display example questions for users."""
        logger.info("Displaying example questions.")
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
        logger.info("Example questions displayed successfully.")

    def display_schema_information(self):
        """Display detailed schema information to the user."""
        logger.info("Displaying schema information.")
        st.markdown("### Data Schema Overview")
        st.markdown(
            """
**Supplier Information:**
- `Supplier Code` (String)
- `Supplier Name` (String)
- `Supplier Qualifications` (Categorical)
- `Supplier Zip Code` (String)

**Acquisition Details:**
- `Fiscal Year` (String)
- `Acquisition Type` (Categorical)
- `Acquisition Method` (Categorical)

**Item Details:**
- `Item Name` (String)
- `Item Description` (String)
- `Quantity` (Integer)
- `Unit Price` (Float)
- `Total Price` (Float)
- `Normalized UNSPSC` (String)

**Date Fields:**
- `Creation Date` (Date)
- `Purchase Date` (Date)

**Department Details:**
- `Department Name` (String)

**Miscellaneous:**
- `CalCard` (Boolean)
- `LPA Number`, `Purchase Order Number`, `Requisition Number` (String)
"""
        )
        logger.info("Schema information displayed successfully.")

    def main(self):
        """Main application logic."""
        logger.info("Starting main application.")
        st.title("Procurement Data Query Assistant")

        st.markdown(
            """
            This application allows you to interact with procurement data from 2012–2015.
            Type your queries in natural language to analyze procurement data.
        """
        )

        # Display data schema
        with st.expander("View Data Schema"):
            self.display_schema_information()

        # Display example questions
        with st.expander("Example Questions to Inspire You"):
            self.display_example_questions()

        # Initialize MongoDB handler
        try:
            logger.info("Initializing MongoDBHandler.")
            mongo_handler = MongoDBHandler(dataset_metadata=self.dataset_metadata)
            logger.info("MongoDBHandler initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MongoDBHandler: {e}")
            st.error(f"Error connecting to MongoDB: {e}")
            return  # Exit the main function if MongoDBHandler fails

        # Initialize LangChain Query Translator
        try:
            openai_api_key = st.secrets["openai_api_key"]
            logger.info("Initializing LangChainQueryTranslator.")
            query_translator = LangChainQueryTranslator(
                openai_api_key=openai_api_key, dataset_metadata=self.dataset_metadata
            )
            logger.info("LangChainQueryTranslator initialized successfully.")
        except KeyError:
            logger.error("OpenAI API key not found in secrets.")
            st.error("OpenAI API key is missing. Please set it in the secrets.")
            return
        except Exception as e:
            logger.error(f"Error initializing LangChainQueryTranslator: {e}")
            st.error(f"Error initializing query translator: {e}")
            return

        # Initialize LangChain Setup for PandasAI
        try:
            logger.info("Setting up LangChain with PandasAI.")
            llm = LangChainSetup.setup_pandasai(openai_api_key=openai_api_key)
            logger.info("LangChain with PandasAI setup successfully.")
        except Exception as e:
            logger.error(f"Error setting up LangChain with PandasAI: {e}")
            st.error(f"Error setting up LangChain: {e}")
            return

        # Input for user query
        query = st.text_input("Enter your query:")

        if st.button("Submit Query") and query:
            logger.info("Submit Query button clicked.")
            with st.spinner("Processing your query..."):
                # Translate natural language query to MongoDB query
                mongo_query = query_translator.translate(
                    user_query=query, dataset_metadata=self.dataset_metadata
                )

                if mongo_query is not None:
                    try:
                        logger.info("Loading data based on MongoDB query.")
                        data = DataLoader.load_data(mongo_handler, mongo_query)

                        if not data.empty:
                            # Initialize PandasAI SmartDataframe with the queried data
                            smart_df = SmartDataframe(
                                data,
                                name="Procurement Dataset",
                                description=self.dataset_metadata,
                                config={
                                    "llm": llm,
                                    "response_parser": StreamlitResponse,
                                    "enable_cache": True,
                                    "verbose": True,
                                    "save_charts": False,
                                    "open_charts": False,
                                    "custom_whitelisted_dependencies": [
                                        "scikit-learn",
                                        "plotly",
                                    ],
                                },
                            )

                            # Process the query with PandasAI
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
                                    st.image(result)
                                else:
                                    st.write(result)
                            elif isinstance(result, dict) and "type" in result:
                                # Handle PandasAI response dictionary
                                if result["type"] == "plot":
                                    if isinstance(result["value"], str):
                                        if result["value"].startswith(
                                            "data:image/png;base64,"
                                        ):
                                            st.image(result["value"].split(",")[1])
                                        else:
                                            st.image(result["value"])
                                elif result["type"] == "dataframe":
                                    st.dataframe(result["value"])
                                else:
                                    st.write(result["value"])
                            else:
                                st.write(result)

                            # Display the generated code in an expander
                            with st.expander("View Generated Code"):
                                st.code(generated_code, language="python")
                        else:
                            st.warning("No data found for the given query.")
                    except Exception as e:
                        logger.error(f"Failed to load data from MongoDB: {e}")
                        st.error(f"Failed to load data from MongoDB: {e}")
                else:
                    logger.error("Failed to translate the query into a MongoDB filter.")
                    st.error("Failed to translate the query into a MongoDB filter.")

        # Optionally, display sample data (10 rows)
        with st.expander("View Sample Data (10 rows)"):
            try:
                logger.info("Fetching sample data from MongoDB.")
                sample_query = {}  # Empty query to fetch any sample
                sample_data = DataLoader.load_data(mongo_handler, sample_query).head(10)
                if not sample_data.empty:
                    st.write(sample_data)
                else:
                    st.write("No sample data available.")
            except Exception as e:
                logger.error(f"Error fetching sample data: {e}")
                st.error(f"Error fetching sample data: {e}")


if __name__ == "__main__":
    try:
        logger.info("Starting application: Applying security bypass.")
        SecurityBypass.bypass_security()  # Initialize security bypass
        app = ProcurementApp()
        app.main()
        logger.info("Application started successfully.")
    except Exception as e:
        logger.critical(f"Unhandled exception in application: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")
