import sys
import json
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm
import streamlit as st

from config import BATCH_SIZE, COLLECTION_NAME, CSV_FILE_PATH, DB_NAME, MONGODB_URI


class MongoDBHandler:
    """Handles MongoDB connection and query execution."""

    def __init__(self, dataset_metadata):
        """Initialize MongoDB connection and parse schema."""
        self.client = self._connect_to_mongodb()
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.schema = self.parse_schema(dataset_metadata)

    def _connect_to_mongodb(self):
        """Establish connection to MongoDB server."""
        try:
            client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
            client.admin.command("ping")
            print("Successfully connected to MongoDB!")
            return client
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            sys.exit(1)

    def parse_schema(self, dataset_metadata):
        """Parse the dataset metadata to extract field types.

        Args:
            dataset_metadata (str): Detailed schema information.

        Returns:
            dict: Mapping of field names to their data types.
        """
        # For simplicity, using a predefined mapping.
        # In a real-world scenario, consider parsing the metadata dynamically.
        return {
            "Supplier Code": "String",
            "Supplier Name": "String",
            "Supplier Qualifications": "Categorical",
            "Supplier Zip Code": "String",
            "Fiscal Year": "String",
            "Acquisition Type": "Categorical",
            "Acquisition Method": "Categorical",
            "Item Name": "String",
            "Item Description": "String",
            "Quantity": "Integer",
            "Unit Price": "Float",
            "Total Price": "Float",
            "Normalized UNSPSC": "String",
            "Creation Date": "Date",
            "Purchase Date": "Date",
            "Department Name": "String",
            "CalCard": "Boolean",
            "LPA Number": "String",
            "Purchase Order Number": "String",
            "Requisition Number": "String",
        }

    def validate_query(self, query):
        """Validate the MongoDB query filter.

        Args:
            query (dict): MongoDB query filter.

        Returns:
            bool: True if valid, False otherwise.
        """
        allowed_fields = set(self.schema.keys())
        allowed_operators = {
            "$gt",
            "$lt",
            "$gte",
            "$lte",
            "$eq",
            "$ne",
            "$in",
            "$nin",
            "$and",
            "$or",
        }

        def recursive_validate(q):
            if isinstance(q, dict):
                for key, value in q.items():
                    if key.startswith("$"):
                        if key not in allowed_operators:
                            return False
                        if isinstance(value, list):
                            for item in value:
                                if not recursive_validate(item):
                                    return False
                        else:
                            if not recursive_validate(value):
                                return False
                    else:
                        if key not in allowed_fields:
                            return False
                        # Validate value based on schema
                        field_type = self.schema.get(key)
                        if isinstance(value, dict):
                            for op in value.keys():
                                if op not in allowed_operators:
                                    return False
                        # Additional type checks can be implemented here
            return True

        return recursive_validate(query)

    def execute_query(self, query):
        """Execute a MongoDB query and return results as a pandas DataFrame.

        Args:
            query (dict): MongoDB query filter.

        Returns:
            pd.DataFrame: Query results.
        """
        try:
            if self.validate_query(query):
                # Handle date range queries if present in the query
                # Convert string dates to datetime objects if necessary
                for date_field in ["Creation Date", "Purchase Date"]:
                    if date_field in query:
                        if isinstance(query[date_field], dict):
                            for op, val in query[date_field].items():
                                if isinstance(val, str):
                                    query[date_field][op] = pd.to_datetime(val)
                cursor = self.collection.find(query)
                data = pd.DataFrame(list(cursor))
                # Optionally, remove the MongoDB-specific '_id' field
                if "_id" in data.columns:
                    data = data.drop(columns=["_id"])
                return data
            else:
                st.error("Invalid query parameters.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return pd.DataFrame()

    def process_csv_in_batches(self, csv_file_path=CSV_FILE_PATH):
        """Process and upload CSV data to MongoDB in batches.

        Args:
            csv_file_path (str): Path to the CSV file to process

        Returns:
            int: Total number of records inserted
        """
        try:
            # Use chunksize parameter to read CSV in chunks
            chunks = pd.read_csv(csv_file_path, chunksize=BATCH_SIZE)
            total_inserted = 0

            # Get total rows for progress bar
            total_rows = sum(1 for _ in open(csv_file_path)) - 1

            with tqdm(total=total_rows, desc="Uploading data") as pbar:
                for chunk in chunks:
                    try:
                        # Clean and validate data if needed
                        chunk = chunk.replace({pd.NA: None})

                        # Convert chunk to list of dictionaries
                        batch_data = chunk.to_dict(orient="records")

                        # Insert batch
                        result = self.collection.insert_many(batch_data, ordered=False)
                        total_inserted += len(result.inserted_ids)
                        pbar.update(len(chunk))

                    except Exception as batch_error:
                        print(f"Error in batch: {batch_error}")
                        continue

            print(f"Successfully inserted {total_inserted} documents")
            return total_inserted

        except Exception as e:
            print(f"Error processing CSV: {e}")
        finally:
            self.client.close()
