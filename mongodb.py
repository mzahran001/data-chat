import sys

import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tqdm import tqdm

from config import BATCH_SIZE, COLLECTION_NAME, CSV_FILE_PATH, DB_NAME, MONGODB_URI


class MongoDBHandler:
    def __init__(self):
        """Initialize MongoDB connection and setup database/collection."""
        self.client = self._connect_to_mongodb()
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

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


if __name__ == "__main__":
    # Create indices before insertion if needed
    # collection.create_index([("field_name", 1)])

    mongo_handler = MongoDBHandler()
    mongo_handler.process_csv_in_batches()
