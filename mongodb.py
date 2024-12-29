from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from tqdm import tqdm
import sys
import streamlit as st

# MongoDB connection URI
uri = f"mongodb+srv://{st.secrets['db_username']}:{st.secrets['db_password']}@cluster0.nzc35.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    exit()

# Define database and collection
db = client["purchase_orders"]
collection = db["order_data"]

# Path to the CSV file
csv_file_path = "PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv"

# Define batch size for processing
BATCH_SIZE = 5000


def process_csv_in_batches(csv_file_path, collection):
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
                    result = collection.insert_many(batch_data, ordered=False)
                    total_inserted += len(result.inserted_ids)
                    pbar.update(len(chunk))

                except Exception as batch_error:
                    print(f"Error in batch: {batch_error}")
                    continue

        print(f"Successfully inserted {total_inserted} documents")

    except Exception as e:
        print(f"Error processing CSV: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    # Create indices before insertion if needed
    # collection.create_index([("field_name", 1)])

    process_csv_in_batches(csv_file_path, collection)
